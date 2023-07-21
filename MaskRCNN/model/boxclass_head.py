# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
from MaskRCNN.performance import print_runtime_shape, print_runtime_tensor

from tensorpack.models import Conv2D, FullyConnected, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized_method

from model.backbone import GroupNorm
from config import config as cfg
from model_box import decode_bbox_target, encode_bbox_target
#from utils.mixed_precision import mixed_precision_scope


@layer_register(log_shape=True)
def boxclass_outputs(feature, num_classes,  seed_gen, class_agnostic_regression=False):
    """
    Args:
        feature: features generated from FasterRCNN head function, Num_boxes x Num_features
        num_classes(int): num_category + 1
        class_agnostic_regression (bool): if True, regression to Num_boxes x 1 x 4

    Returns:
        cls_logits: Num_boxes x Num_classes classification logits
        reg_logits: Num_boxes x num_classes x 4 or Num_boxes x 2 x 4 if class agnostic
    """
    classification = FullyConnected(
        'class', feature, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed_gen.next()))
    num_classes_for_box = 1 if class_agnostic_regression else num_classes
    box_regression = FullyConnected(
        'box', feature, num_classes_for_box * 4,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001, seed=seed_gen.next()))
    box_regression = tf.reshape(box_regression, [-1, num_classes_for_box, 4], name='output_box')
    return classification, box_regression



@under_name_scope()
def boxclass_losses(labels_gt, labels_pred, fg_boxes_gt, fg_boxes_pred):
    """
    Args:
        labels_gt: Num_boxes
        labels_pred:  Num_boxes x Num_classes
        fg_boxes_gt: Num_fg_boxes x 4, encoded
        fg_boxes_pred: Num_boxes x Num_classes x 4 (default) or Num_boxes x 1 x 4 (class agnostic)

    Returns:
        label_loss, box_loss
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_gt, logits=labels_pred)
    label_loss = tf.math.reduce_mean(label_loss, name='label_loss')

    fg_inds = tf.where(labels_gt > 0)[:, 0]
    fg_labels = tf.gather(labels_gt, fg_inds)
    
    #fg_labels = print_runtime_tensor("fg_labels", fg_labels)

    num_fg = tf.size(fg_inds, out_type=tf.int64)
    empty_fg = tf.equal(num_fg, 0)
    if int(fg_boxes_pred.shape[1]) > 1:
        indices = tf.stack(
            [tf.range(num_fg), fg_labels], axis=1)  # #fgx2
        fg_boxes_pred = tf.gather_nd(fg_boxes_pred, indices)
    else:
        fg_boxes_pred = tf.reshape(fg_boxes_pred, [-1, 4])

    with tf.name_scope('label_metrics'):
        prediction = tf.argmax(labels_pred, axis=1, name='label_prediction')
        correct = tf.cast(tf.math.equal(prediction, labels_gt), tf.float32)  # boolean/integer gather is unavailable on GPU
        accuracy = tf.math.reduce_mean(correct, name='accuracy')
        fg_label_pred = tf.argmax(tf.gather(labels_pred, fg_inds), axis=1)
        num_zero = tf.reduce_sum(tf.cast(tf.equal(fg_label_pred, 0), tf.int64), name='num_zero')
        false_negative = tf.where(
            empty_fg, 0., tf.cast(tf.truediv(num_zero, num_fg), tf.float32), name='false_negative')
        fg_accuracy = tf.where(
            empty_fg, 0., tf.math.reduce_mean(tf.gather(correct, fg_inds)), name='fg_accuracy')

        #fg_accuracy = print_runtime_tensor("fg_accuracy", fg_accuracy)

    box_loss = tf.compat.v1.losses.huber_loss(fg_boxes_gt, fg_boxes_pred, delta=1.0, reduction='none')
    box_loss = tf.math.reduce_sum(box_loss)
    box_loss = tf.truediv(box_loss, tf.cast(tf.shape(labels_gt)[0], tf.float32), name='box_loss')
    
    #box_loss = print_runtime_tensor("box_loss", box_loss)
    #label_loss = print_runtime_tensor("label_loss", label_loss)
    #false_negative = print_runtime_tensor("false_negative", false_negative)

    add_moving_summary(label_loss, box_loss, accuracy,
                       fg_accuracy, false_negative, tf.cast(num_fg, tf.float32, name='num_fg_label'))
    return [label_loss, box_loss]


def boxclass_predictions(boxes, scores):
    """
    Generate final results from predictions of all proposals.

    Args:
        boxes: N X #class X 4 floatbox in float32
        scores: N X #class

    Returns:
        boxes: K X 4 (x1,y1,x2,y2)
        scores: K
        labels: K
    """
    assert boxes.shape[1] == cfg.DATA.NUM_CLASS
    assert scores.shape[1] == cfg.DATA.NUM_CLASS
    boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # #class X N X 4
    scores = tf.transpose(scores[:, 1:], [1, 0])  # #class X N

    max_coord = tf.reduce_max(boxes)
    filtered_indices = tf.where(scores > cfg.TEST.RESULT_SCORE_THRESH)  # Fx2
    filtered_boxes = tf.gather_nd(boxes, filtered_indices)  # Fx4
    filtered_scores = tf.gather_nd(scores, filtered_indices)  # F,
    cls_per_box = tf.slice(filtered_indices, [0, 0], [-1, 1])
    offsets = tf.cast(cls_per_box, tf.float32) * (max_coord + 1)  # F,1
    
    selection = tf.image.non_max_suppression(
        filtered_boxes + offsets,
        filtered_scores,
        cfg.TEST.RESULTS_PER_IM,
        cfg.TEST.FRCNN_NMS_THRESH)
    filtered_selection = tf.gather(filtered_indices, selection)
    _, box_ids = tf.unstack(filtered_selection, axis=1)
    final_scores = tf.gather(filtered_scores, selection)
    final_labels = tf.add(tf.gather(cls_per_box[:, 0], selection), 1)
    final_boxes = tf.gather(filtered_boxes, selection)
    return final_boxes, final_scores, final_labels, box_ids



"""
FastRCNN heads for FPN:
"""


@layer_register(log_shape=True)
def boxclass_2fc_head(feature, seed_gen):
    """
    Fully connected layer for the class and box branch

    Args:
        feature map: The roi feature map, Num_boxes x  H_roi x W_roi x Num_channels

    Returns:
        2D head feature: Num_boxes x Num_features
    """
    dim = cfg.FPN.BOXCLASS_FC_HEAD_DIM
    
    init = tf.keras.initializers.VarianceScaling(scale=1.0, seed=seed_gen.next())
    hidden = FullyConnected('fc6', feature, dim, kernel_initializer=init, activation=tf.nn.relu)
    hidden = FullyConnected('fc7', hidden, dim, kernel_initializer=init, activation=tf.nn.relu)

    return hidden


@layer_register(log_shape=True)
def boxclass_Xconv1fc_head(feature, seed_gen, num_convs, norm=None):
    """
    Args:
        feature (NCHW):
        num_classes(int): num_category + 1
        num_convs (int): number of conv layers
        norm (str or None): either None or 'GN'

    Returns:
        2D head feature
    """
    assert norm in [None, 'GN'], norm
    l = feature
    with argscope(Conv2D, data_format='channels_first',
                  kernel_initializer=tf.keras.initializers.VarianceScaling (
                      scale=2.0, mode='fan_out',
                      distribution='untruncated_normal',
                      seed=seed_gen.next())):
        for k in range(num_convs):
            l = Conv2D('conv{}'.format(k), l, cfg.FPN.BOXCLASS_CONV_HEAD_DIM, 3, activation=tf.nn.relu, seed=seed_gen.next())
            if norm is not None:
                l = GroupNorm('gn{}'.format(k), l)
        l = FullyConnected('fc', l, cfg.FPN.BOXCLASS_FC_HEAD_DIM,
                           kernel_initializer=tf.keras.initializers.VarianceScaling(seed=seed_gen.next()), 
                           activation=tf.nn.relu, seed=seed_gen.next())
    return l


def boxclass_4conv1fc_head(*args, **kwargs):
    return boxclass_Xconv1fc_head(*args, num_convs=4, **kwargs)


def boxclass_4conv1fc_gn_head(*args, **kwargs):
    return boxclass_Xconv1fc_head(*args, num_convs=4, norm='GN', **kwargs)




class BoxClassHead(object):
    """
    A class to process & decode inputs/outputs of a fastrcnn classification+regression head.
    """
    def __init__(self,
                 boxes_pred,
                 labels_pred,
                 bbox_regression_weights,
                 proposal_rois):
        """
        Args:
            boxes_pred: BS X Num_boxes x Num_classes x 4 (default) or Num_boxes x 1 x 4 (class agnostic), the output of the head
            labels_pred: BS X Num_boxes x Num_classes, the output of the head
            bbox_regression_weights: a 4 element tensor
            proposal_rois: BS X Num_boxs x 4
        """
        self.boxes_pred = boxes_pred
        self.labels_pred = labels_pred
        self.bbox_regression_weights = bbox_regression_weights
        self.proposal_rois = proposal_rois
        self.training_info_available = False

    def add_training_info(self, proposal_boxes_gt, proposal_labels_gt):
        """
        Args:
            proposal_boxes_gt: BS x Num_samples x 4
            proposal_labels_gt: BS X Num_samples
        """
        self.proposal_boxes_gt = proposal_boxes_gt
        self.proposal_labels_gt = proposal_labels_gt
        self.training_info_available = True

    @memoized_method
    def losses(self):

        assert self.training_info_available, "In order to calculate losses, we need to know GT info, but " \
                                             "add_training_info was never called"


        proposal_rois = tf.reshape(self.proposal_rois, [-1, 4])
        proposal_boxes_gt = tf.reshape(self.proposal_boxes_gt, [-1, 4])
        
        boxes_pred_shape = tf.shape(self.boxes_pred)
        boxes_pred = tf.reshape(self.boxes_pred, [-1, boxes_pred_shape[-2], 4])
        
        proposal_labels_gt = tf.reshape(self.proposal_labels_gt, [-1])
        
        labels_pred_shape = tf.shape(self.labels_pred)
        labels_pred = tf.reshape(self.labels_pred, [-1, labels_pred_shape[-1]])
        
        fg_proposal_indices = tf.reshape(tf.where(proposal_labels_gt > 0), [-1])
        fg_proposal_rois = tf.gather(proposal_rois, fg_proposal_indices) # NumFG x 4
        fg_proposal_boxes_gt = tf.gather(proposal_boxes_gt, fg_proposal_indices) # NumFG x 4

        encoded_fg_boxes_gt = encode_bbox_target(fg_proposal_boxes_gt, fg_proposal_rois)
        encoded_fg_boxes_gt = encoded_fg_boxes_gt * self.bbox_regression_weights
        fg_boxes_pred = tf.gather(boxes_pred, fg_proposal_indices)
        
        return boxclass_losses(
            proposal_labels_gt,
            labels_pred,
            encoded_fg_boxes_gt,
            fg_boxes_pred
        )

    @memoized_method
    def decoded_output_boxes_batch(self):
        """ Returns: BS X N x #class x 4 """
        anchors = tf.tile(tf.expand_dims(self.proposal_rois, 2),
                          [1, 1, cfg.DATA.NUM_CLASS, 1])  # BS X N x #class x 4
        decoded_boxes = decode_bbox_target(
                self.boxes_pred / self.bbox_regression_weights,
                anchors
        )
        return decoded_boxes


    @memoized_method
    def decoded_output_boxes(self):
        """ Returns: N x #class x 4 """
        anchors = tf.tile(tf.expand_dims(self.proposal_rois, 1),
                      [1, cfg.DATA.NUM_CLASS, 1])   # N x #class x 4
        decoded_boxes = decode_bbox_target(
            self.boxes_pred / self.bbox_regression_weights,
            anchors
        )
        return decoded_boxes


    @memoized_method
    def output_scores(self, name=None):
        """ Returns: BS X N x #class scores, summed to one for each box."""
        return tf.nn.softmax(self.labels_pred, name=name)
