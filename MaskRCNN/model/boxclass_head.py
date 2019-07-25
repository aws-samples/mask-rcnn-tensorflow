# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
# File: model.py

import tensorflow as tf

from tensorpack.models import Conv2D, FullyConnected, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized_method

from model.backbone import GroupNorm
from config import config as cfg
from model_box import decode_bbox_target, encode_bbox_target
from utils.mixed_precision import mixed_precision_scope


@layer_register(log_shape=True)
def boxclass_outputs(feature, num_classes, seed_gen, class_agnostic_regression=False):
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
def boxclass_losses(labels, label_logits, fg_boxes, fg_box_logits):
    """
    Args:
        labels: Num_boxes
        label_logits:  Num_boxes x Num_classes
        fg_boxes: Num_fg_boxes x 4, encoded
        fg_box_logits: Num_boxes x Num_classes x 4 (default) or Num_boxes x 1 x 4 (class agnostic)

    Returns:
        label_loss, box_loss
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')

    fg_inds = tf.where(labels > 0)[:, 0]
    fg_labels = tf.gather(labels, fg_inds)
    num_fg = tf.size(fg_inds, out_type=tf.int64)
    empty_fg = tf.equal(num_fg, 0)
    if int(fg_box_logits.shape[1]) > 1:
        indices = tf.stack(
            [tf.range(num_fg), fg_labels], axis=1)  # #fgx2
        fg_box_logits = tf.gather_nd(fg_box_logits, indices)
    else:
        fg_box_logits = tf.reshape(fg_box_logits, [-1, 4])

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
        correct = tf.cast(tf.equal(prediction, labels), tf.float32)  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name='accuracy')
        fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
        num_zero = tf.reduce_sum(tf.cast(tf.equal(fg_label_pred, 0), tf.int64), name='num_zero')
        false_negative = tf.where(
            empty_fg, 0., tf.cast(tf.truediv(num_zero, num_fg), tf.float32), name='false_negative')
        fg_accuracy = tf.where(
            empty_fg, 0., tf.reduce_mean(tf.gather(correct, fg_inds)), name='fg_accuracy')

    box_loss = tf.losses.huber_loss(
        fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)
    box_loss = tf.truediv(
        box_loss, tf.cast(tf.shape(labels)[0], tf.float32), name='box_loss')

    add_moving_summary(label_loss, box_loss, accuracy,
                       fg_accuracy, false_negative, tf.cast(num_fg, tf.float32, name='num_fg_label'))
    return [label_loss, box_loss]


@under_name_scope()
def boxclass_predictions(boxes, scores):
    """
    Generate final results from predictions of all proposals.

    Args:
        boxes: n#classx4 floatbox in float32
        scores: nx#class

    Returns:
        boxes: Kx4
        scores: K
        labels: K
    """
    assert boxes.shape[1] == cfg.DATA.NUM_CLASS
    assert scores.shape[1] == cfg.DATA.NUM_CLASS
    boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # #catxnx4
    scores = tf.transpose(scores[:, 1:], [1, 0])  # #catxn

    def f(X):
        """
        prob: n probabilities
        box: nx4 boxes

        Returns: n boolean, the selection
        """
        prob, box = X
        output_shape = tf.shape(prob, out_type=tf.int64)
        # filter by score threshold
        ids = tf.reshape(tf.where(prob > cfg.TEST.RESULT_SCORE_THRESH), [-1])
        prob = tf.gather(prob, ids)
        box = tf.gather(box, ids)
        # NMS within each class
        #selection = non_max_suppression_custom(
            #box, prob, cfg.TEST.RESULTS_PER_IM, cfg.TEST.FRCNN_NMS_THRESH)
        selection = tf.image.non_max_suppression(
            box, prob, cfg.TEST.RESULTS_PER_IM, cfg.TEST.FRCNN_NMS_THRESH)
        selection = tf.gather(ids, selection)

        if get_tf_version_tuple() >= (1, 13):
            sorted_selection = tf.sort(selection, direction='ASCENDING')
            mask = tf.sparse.SparseTensor(indices=tf.expand_dims(sorted_selection, 1),
                                          values=tf.ones_like(sorted_selection, dtype=tf.bool),
                                          dense_shape=output_shape)
            mask = tf.sparse.to_dense(mask, default_value=False)
        else:
            # this function is deprecated by TF
            sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
            mask = tf.sparse_to_dense(
                sparse_indices=sorted_selection,
                output_shape=output_shape,
                sparse_values=True,
                default_value=False)
        return mask

    # TF bug in version 1.11, 1.12: https://github.com/tensorflow/tensorflow/issues/22750
    buggy_tf = get_tf_version_tuple() in [(1, 11), (1, 12)]
    masks = tf.map_fn(f, (scores, boxes), dtype=tf.bool,
                      parallel_iterations=1 if buggy_tf else 10)     # #cat x N
    selected_indices = tf.where(masks)  # #selection x 2, each is (cat_id, box_id)
    scores = tf.boolean_mask(scores, masks)

    # filter again by sorting scores
    topk_scores, topk_indices = tf.nn.top_k(
        scores,
        tf.minimum(cfg.TEST.RESULTS_PER_IM, tf.size(scores)),
        sorted=False)
    filtered_selection = tf.gather(selected_indices, topk_indices)
    cat_ids, box_ids = tf.unstack(filtered_selection, axis=1)

    final_scores = tf.identity(topk_scores, name='scores')
    final_labels = tf.add(cat_ids, 1, name='labels')
    final_ids = tf.stack([cat_ids, box_ids], axis=1, name='all_ids')
    final_boxes = tf.gather_nd(boxes, final_ids, name='boxes')
    return final_boxes, final_scores, final_labels, box_ids



"""
FastRCNN heads for FPN:
"""


@layer_register(log_shape=True)
def boxclass_2fc_head(feature, seed_gen, fp16=False):
    """
    Fully connected layer for the class and box branch

    Args:
        feature map: The roi feature map, Num_boxes x Num_channels x H_roi x W_roi

    Returns:
        2D head feature: Num_boxes x Num_features
    """
    dim = cfg.FPN.BOXCLASS_FC_HEAD_DIM
    if fp16:
        feature = tf.cast(feature, tf.float16)

    with mixed_precision_scope(mixed=fp16):
        init = tf.variance_scaling_initializer(dtype=tf.float16 if fp16 else tf.float32, seed=seed_gen.next())
        hidden = FullyConnected('fc6', feature, dim, kernel_initializer=init, activation=tf.nn.relu)
        hidden = FullyConnected('fc7', hidden, dim, kernel_initializer=init, activation=tf.nn.relu)

    if fp16:
        hidden = tf.cast(hidden, tf.float32)

    return hidden


@layer_register(log_shape=True)
def boxclass_Xconv1fc_head(feature, num_convs, norm=None):
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
                  kernel_initializer=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_out',
                      distribution='untruncated_normal' if get_tf_version_tuple() >= (1, 12) else 'normal')):
        for k in range(num_convs):
            l = Conv2D('conv{}'.format(k), l, cfg.FPN.BOXCLASS_CONV_HEAD_DIM, 3, activation=tf.nn.relu)
            if norm is not None:
                l = GroupNorm('gn{}'.format(k), l)
        l = FullyConnected('fc', l, cfg.FPN.BOXCLASS_FC_HEAD_DIM,
                           kernel_initializer=tf.variance_scaling_initializer(), activation=tf.nn.relu)
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
                 box_logits,
                 label_logits,
                 bbox_regression_weights,
                 prepadding_gt_counts,
                 proposal_boxes):
        """
        Args:
            box_logits: Num_boxes x Num_classes x 4 (default) or Num_boxes x 1 x 4 (class agnostic), the output of the head
            label_logits: Num_boxes x Num_classes, the output of the head
            bbox_regression_weights: a 4 element tensor
            prepadding_gt_counts: The original gt box number before padding for each image
            proposal_boxes: Num_boxs x 5
        """
        self.box_logits = box_logits
        self.label_logits = label_logits

        self.bbox_regression_weights = bbox_regression_weights
        self.prepadding_gt_counts = prepadding_gt_counts

        self.proposal_boxes = proposal_boxes

        self._bbox_class_agnostic = int(box_logits.shape[1]) == 1

        self.training_info_available = False


    def add_training_info(self,
                          gt_boxes,
                          proposal_labels,
                          proposal_fg_inds,
                          proposal_fg_boxes,
                          proposal_fg_labels,
                          proposal_gt_id_for_each_fg):
        """
        Args:
            gt_boxes: BS x Num_gt_boxes x 4
            proposal_labels: 1-D Num_boxes
            proposal_fg_inds: 1-D Num_fg_boxes
            proposal_fg_boxes: Num_fg_boxs x 5
            proposal_fg_labels: 1-D Num_fg_boxes
            proposal_gt_id_for_each_fg: indices for matching GT of each foreground box, BS x [Num_fg_boxes_per_image]
        """

        self.gt_boxes = gt_boxes
        self.proposal_labels = proposal_labels
        self.proposal_fg_inds = proposal_fg_inds
        self.proposal_fg_boxes = proposal_fg_boxes
        self.proposal_fg_labels = proposal_fg_labels
        self.proposal_gt_id_for_each_fg = proposal_gt_id_for_each_fg

        self.training_info_available = True


    @memoized_method
    def losses(self, batch_size_per_gpu, shortcut=False):

        assert self.training_info_available, "In order to calculate losses, we need to know GT info, but " \
                                             "add_training_info was never called"

        if shortcut:
            proposal_label_loss = tf.cast(tf.reduce_mean(self.proposal_labels), dtype=tf.float32)
            proposal_boxes_loss = tf.cast(tf.reduce_mean(self.proposal_boxes), dtype=tf.float32)
            proposal_fg_boxes_loss = tf.cast(tf.reduce_mean(self.proposal_fg_boxes), dtype=tf.float32)
            gt_box_loss = tf.cast(tf.reduce_mean(self.gt_boxes), dtype=tf.float32)

            bbox_reg_loss = tf.cast(tf.reduce_mean(self.bbox_regression_weights), dtype=tf.float32)
            label_logit_loss = tf.cast(tf.reduce_mean(self.label_logits), dtype=tf.float32)

            total_loss = proposal_label_loss + proposal_boxes_loss + proposal_fg_boxes_loss + gt_box_loss \
                         + bbox_reg_loss + label_logit_loss
            return [total_loss]

        all_labels = []
        all_label_logits = []
        all_encoded_fg_gt_boxes = []
        all_fg_box_logits = []
        for i in range(batch_size_per_gpu):

            single_image_fg_inds_wrt_gt = self.proposal_gt_id_for_each_fg[i]

            single_image_gt_boxes = self.gt_boxes[i, :self.prepadding_gt_counts[i], :] # NumGT x 4
            gt_for_each_fg = tf.gather(single_image_gt_boxes, single_image_fg_inds_wrt_gt) # NumFG x 4
            single_image_fg_boxes_indices = tf.where(tf.equal(self.proposal_fg_boxes[:, 0], i))
            single_image_fg_boxes_indices = tf.squeeze(single_image_fg_boxes_indices, axis=1)

            single_image_fg_boxes = tf.gather(self.proposal_fg_boxes, single_image_fg_boxes_indices) # NumFG x 5
            single_image_fg_boxes = single_image_fg_boxes[:, 1:]  # NumFG x 4

            encoded_fg_gt_boxes = encode_bbox_target(gt_for_each_fg, single_image_fg_boxes) * self.bbox_regression_weights

            single_image_box_indices = tf.squeeze(tf.where(tf.equal(self.proposal_boxes[:, 0], i)), axis=1)
            single_image_labels = tf.gather(self.proposal_labels, single_image_box_indices) # Vector len N
            single_image_label_logits = tf.gather(self.label_logits, single_image_box_indices)

            single_image_fg_box_logits_indices = tf.gather(self.proposal_fg_inds, single_image_fg_boxes_indices)
            single_image_fg_box_logits = tf.gather(self.box_logits, single_image_fg_box_logits_indices)

            all_labels.append(single_image_labels)
            all_label_logits.append(single_image_label_logits)
            all_encoded_fg_gt_boxes.append(encoded_fg_gt_boxes)
            all_fg_box_logits.append(single_image_fg_box_logits)



        return boxclass_losses(
            tf.concat(all_labels, axis=0),
            tf.concat(all_label_logits, axis=0),
            tf.concat(all_encoded_fg_gt_boxes, axis=0),
            tf.concat(all_fg_box_logits, axis=0)
        )

    @memoized_method
    def decoded_output_boxes_batch(self):
        """ Returns: N x #class x 4 """
        batch_ids, nobatch_proposal_boxes = tf.split(self.proposal_boxes, [1, 4], 1)
        anchors = tf.tile(tf.expand_dims(nobatch_proposal_boxes, 1),
                          [1, cfg.DATA.NUM_CLASS, 1])  # N x #class x 4
        decoded_boxes = decode_bbox_target(
                self.box_logits / self.bbox_regression_weights,
                anchors
        )
        return decoded_boxes, tf.reshape(batch_ids, [-1])


    @memoized_method
    def decoded_output_boxes(self):
        """ Returns: N x #class x 4 """
        anchors = tf.tile(tf.expand_dims(self.proposal_boxes, 1),
                      [1, cfg.DATA.NUM_CLASS, 1])   # N x #class x 4
        decoded_boxes = decode_bbox_target(
            self.box_logits / self.bbox_regression_weights,
            anchors
        )
        return decoded_boxes


    @memoized_method
    def output_scores(self, name=None):
        """ Returns: N x #class scores, summed to one for each box."""
        return tf.nn.softmax(self.label_logits, name=name)
