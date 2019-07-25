# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack import get_current_tower_context
from tensorpack.models import Conv2D, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope, under_name_scope
from tensorpack.tfutils.summary import add_moving_summary

from config import config as cfg
from model_box import clip_boxes
from utils.mixed_precision import mixed_precision_scope

@layer_register(log_shape=True)
@auto_reuse_variable_scope
def rpn_head(featuremap, channel, num_anchors, seed_gen, fp16=False):
    """
    The RPN head that takes the feature map from the FPN and outputs bounding box logits.
    For every pixel on the feature maps, there are a certain number of anchors.
    The output will be:
    label logits: indicate whether there is an object for a certain anchor in one pixel
    box logits: The encoded box logits from fast-rcnn paper https://arxiv.org/abs/1506.01497
                page 5, in order to be consistent with the ground truth encoded boxes

    Args:
        featuremap: feature map for a single FPN layer, i.e. one from P23456, BS x NumChannel x H_feature x W_feature
        channel: NumChannel of the feature map, scalar, default 256
        num_anchors(NA): # of anchors for each pixel in the current feature map, scalar, default 3
    Returns:
        label_logits: BS x H_feature x W_feature x NA
        box_logits: BS x (NA * 4) x H_feature x W_feature, encoded
    """
    if fp16:
        featuremap = tf.cast(featuremap, tf.float16)

    with mixed_precision_scope(mixed=fp16):
        with argscope(Conv2D, data_format='channels_first',
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed_gen.next())):
            hidden = Conv2D('conv0', featuremap, channel, 3, activation=tf.nn.relu, seed=seed_gen.next())
            # BS x NumChannel x H_feature x W_feature
            label_logits = Conv2D('class', hidden, num_anchors, 1, seed=seed_gen.next())
            # BS x NA x H_feature x W_feature
            box_logits = Conv2D('box', hidden, 4 * num_anchors, 1, seed=seed_gen.next())
            # BS x (NA*4) x H_feature x W_feature

            label_logits = tf.transpose(label_logits, [0, 2, 3, 1])  # BS x H_feature x W_feature x NA

    if fp16:
        label_logits = tf.cast(label_logits, tf.float32)
        box_logits = tf.cast(box_logits, tf.float32)

    return label_logits, box_logits



@under_name_scope()
def rpn_losses(anchor_labels, anchor_boxes, label_logits, box_logits):
    """
    Calculate the rpn loss for one FPN layer for a single image.
    The ground truth(GT) anchor labels and anchor boxes has been preprocessed to fit
    the dimensions of FPN feature map. The GT boxes are encoded from fast-rcnn paper
    https://arxiv.org/abs/1506.01497 page 5.

    Args:
        anchor_labels: GT anchor labels, H_feature x W_feature x NA
        anchor_boxes: GT boxes for each anchor, H_feature x W_feature x NA x 4, encoded
        label_logits: label logits from the rpn head, H_feature x W_feature x NA
        box_logits: box logits from the rpn head, H_feature x W_feature x NA x 4
    Returns:
        label_loss, box_loss
    """
    with tf.device('/cpu:0'):
        valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))
        pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
        nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name='num_valid_anchor')
        nr_pos = tf.identity(tf.count_nonzero(pos_mask, dtype=tf.int32), name='num_pos_anchor')
        # nr_pos is guaranteed >0 in C4. But in FPN. even nr_valid could be 0.

        valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)
    valid_label_logits = tf.boolean_mask(label_logits, valid_mask)

    # with tf.name_scope('label_metrics'):
    #     valid_label_prob = tf.nn.sigmoid(valid_label_logits)
    #     summaries = []
    #     with tf.device('/cpu:0'):
    #         for th in [0.5, 0.2, 0.1]:
    #             valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
    #             nr_pos_prediction = tf.reduce_sum(valid_prediction, name='num_pos_prediction')
    #             pos_prediction_corr = tf.count_nonzero(
    #                 tf.logical_and(
    #                     valid_label_prob > th,
    #                     tf.equal(valid_prediction, valid_anchor_labels)),
    #                 dtype=tf.int32)
    #             placeholder = 0.5   # A small value will make summaries appear lower.
    #             recall = tf.cast(tf.truediv(pos_prediction_corr, nr_pos), tf.float32)
    #             recall = tf.where(tf.equal(nr_pos, 0), placeholder, recall, name='recall_th{}'.format(th))
    #             precision = tf.cast(tf.truediv(pos_prediction_corr, nr_pos_prediction), tf.float32)
    #             precision = tf.where(tf.equal(nr_pos_prediction, 0),
    #                                  placeholder, precision, name='precision_th{}'.format(th))
    #             summaries.extend([precision, recall])
    #     add_moving_summary(*summaries)

    # Per-level loss summaries in FPN may appear lower due to the use of a small placeholder.
    # But the total RPN loss will be fine.  TODO make the summary op smarter
    placeholder = 0.
    label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(valid_anchor_labels, tf.float32), logits=valid_label_logits)
    label_loss = tf.reduce_sum(label_loss) * (1. / cfg.RPN.BATCH_PER_IM)
    label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name='label_loss')

    pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
    pos_box_logits = tf.boolean_mask(box_logits, pos_mask)
    delta = 1.0 / 9
    box_loss = tf.losses.huber_loss(
        pos_anchor_boxes, pos_box_logits, delta=delta,
        reduction=tf.losses.Reduction.SUM) / delta
    box_loss = box_loss * (1. / cfg.RPN.BATCH_PER_IM)
    box_loss = tf.where(tf.equal(nr_pos, 0), placeholder, box_loss, name='box_loss')

    # add_moving_summary(label_loss, box_loss, nr_valid, nr_pos)
    return [label_loss, box_loss]


def multilevel_rpn_losses(multilevel_anchors, multilevel_label_logits, multilevel_box_logits):
    """
    Calculate the rpn loss for all FPN layers for a single image.

    Args:
        multilevel_anchors: #lvl RPNAnchors
        multilevel_label_logits: [H_feature x W_feature x NA] * Num_levels
        multilevel_box_logits: [H_feature x W_feature x NA x 4] * Num_levels
    Returns:
        label_loss, box_loss
    """
    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_anchors) == num_lvl
    assert len(multilevel_label_logits) == num_lvl
    assert len(multilevel_box_logits) == num_lvl

    losses = []
    with tf.name_scope('single_image_rpn_losses'):
        for lvl in range(num_lvl):
            anchors = multilevel_anchors[lvl]
            label_loss, box_loss = rpn_losses(
                anchors.gt_labels, anchors.encoded_gt_boxes(),
                multilevel_label_logits[lvl], multilevel_box_logits[lvl],
                name_scope='level{}'.format(lvl + 2))
            losses.extend([label_loss, box_loss])

        total_label_loss = tf.add_n(losses[::2])
        total_box_loss = tf.add_n(losses[1::2])
    return [total_label_loss, total_box_loss]



@under_name_scope()
def generate_fpn_proposals(multilevel_anchor_boxes,
                           multilevel_box_logits,
                           multilevel_label_logits,
                           orig_image_dims,
                           batch_size):
    """
    Generating the rois from the box logits and pick K with top label scores as
    the box proposals.

    Args:
        multilevel_box_logits:      #lvl [ BS x (NA * 4) x H_feature x W_feature ] boxes
        multilevel_label_logits:    #lvl [ BS x H_feature x W_feature x NA ] tensors
        orig_image_dimensions: Original (prepadding) image dimensions (h,w,c)   BS x 3
    Returns:
        boxes: K x 5 float
        scores:  1-D, K (logits)
    """
    prefix = "model_fpn.generate_fpn_proposals"
    bug_prefix = "GEN_PROPOSALS_BUG fpn"
    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_label_logits) == num_lvl
    orig_images_hw = orig_image_dims[:, :2]

    training = get_current_tower_context().is_training
    all_boxes = []
    all_scores = []
    if cfg.FPN.PROPOSAL_MODE == 'Level':
        fpn_nms_topk = cfg.RPN.TRAIN_PER_LEVEL_NMS_TOPK*batch_size if training else cfg.RPN.TEST_PER_LEVEL_NMS_TOPK
        for lvl in range(num_lvl):
            with tf.name_scope(f'Lvl{lvl}'):
                im_info = tf.cast(orig_images_hw, tf.float32)

                scores = multilevel_label_logits[lvl] # BS x H_feature x W_featurex NA
                bbox_deltas = tf.transpose(multilevel_box_logits[lvl],[0, 2, 3, 1]) #BS x H_feature x W_feature x (NA * 4)

                single_level_anchor_boxes = multilevel_anchor_boxes[lvl]
                single_level_anchor_boxes = tf.reshape(single_level_anchor_boxes, (-1, 4))


                # # This is a custom tensorflow op that translates the bbox deltas into bounding box coordinates
                # and then runs NMS. See CODEBASE.md for more info
                #
                # roi: (# boxes for a single level) x 5, the 5 colunms arranged as: batch_index, x_1, y_1, x_2, y_2
                # rois_probs: 1-D, # boxes for a single level
                rois, rois_probs = tf.generate_bounding_box_proposals(scores,
                                                                   bbox_deltas,
                                                                   im_info,
                                                                   single_level_anchor_boxes,
                                                                   spatial_scale=1.0 / cfg.FPN.ANCHOR_STRIDES[lvl],
                                                                   pre_nms_topn=fpn_nms_topk,
                                                                   post_nms_topn=fpn_nms_topk,
                                                                   nms_threshold=cfg.RPN.PROPOSAL_NMS_THRESH,
                                                                   min_size=cfg.RPN.MIN_SIZE)
                # rois_probs = print_runtime_shape(f'rois_probs, lvl {lvl}', rois_probs, prefix=bug_prefix)
                all_boxes.append(rois)
                all_scores.append(rois_probs)

        proposal_boxes = tf.concat(all_boxes, axis=0)  # Num_all_rois x 5
        proposal_boxes = tf.reshape(proposal_boxes, [-1, 5]) # Num_all_rois x 5

        proposal_scores = tf.concat(all_scores, axis=0)  # 1-D Num_all_rois
        proposal_scores = tf.reshape(proposal_scores, [-1])  # 1-D Num_all_rois

        proposal_topk = tf.minimum(tf.size(proposal_scores), fpn_nms_topk)
        proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=False)
        proposal_boxes = tf.gather(proposal_boxes, topk_indices) # K x 5

    else:
        raise RuntimeError("Only level-wise predictions are supported with batches")

    return tf.stop_gradient(proposal_boxes, name='boxes'), \
        tf.stop_gradient(proposal_scores, name='scores')




@under_name_scope()
def generate_fpn_proposals_topk_per_image(multilevel_anchor_boxes,
                                          multilevel_box_logits,
                                          multilevel_label_logits,
                                          orig_image_dims,
                                          batch_size):
    """
    Args:
        multilevel_box_logits:      #lvl [ BS x (NAx4) x H x W ] boxes
        multilevel_label_logits:    #lvl [ BS x H x W x A ] tensors
        orig_image_dimensions: Original (prepadding) image dimensions (h,w,c)   BS x 3
    Returns:
        boxes: K x 5 float
        scores:  (#lvl x BS x K) vector       (logits)
    """

    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_label_logits) == num_lvl
    orig_images_hw = orig_image_dims[:, :2]

    training = get_current_tower_context().is_training
    all_boxes = []
    all_scores = []
    if cfg.FPN.PROPOSAL_MODE == 'Level':
        fpn_nms_topk = cfg.RPN.TRAIN_PER_LEVEL_NMS_TOPK if training else cfg.RPN.TEST_PER_LEVEL_NMS_TOPK
        boxes_list = []
        scores_list = []

        bs = batch_size if training else 1

        for i in range(bs):
            all_boxes = []
            all_scores = []
            for lvl in range(num_lvl):
                with tf.name_scope(f'Lvl{lvl}'):
                    im_info = tf.cast(orig_images_hw[i:(i + 1)], tf.float32)
                    # h, w

                    scores = multilevel_label_logits[lvl][i:(i + 1)]
                    bbox_deltas = tf.transpose(multilevel_box_logits[lvl][i:(i + 1)], [0, 2, 3, 1])

                    single_level_anchor_boxes = multilevel_anchor_boxes[lvl]
                    single_level_anchor_boxes = tf.reshape(single_level_anchor_boxes, (-1, 4))

                    # https://caffe2.ai/docs/operators-catalogue.html#generateproposals
                    rois, rois_probs = tf.generate_bounding_box_proposals(scores,
                                                                          bbox_deltas,
                                                                          im_info,
                                                                          single_level_anchor_boxes,
                                                                          spatial_scale=1.0 / cfg.FPN.ANCHOR_STRIDES[
                                                                              lvl],
                                                                          pre_nms_topn=fpn_nms_topk,
                                                                          post_nms_topn=fpn_nms_topk,
                                                                          nms_threshold=cfg.RPN.PROPOSAL_NMS_THRESH,
                                                                          min_size=cfg.RPN.MIN_SIZE)

                    # rois_probs = print_runtime_shape(f'rois_probs, lvl {lvl}', rois_probs, prefix=bug_prefix)
                    all_boxes.append(tf.concat((i + rois[:, :1], rois[:, 1:]), axis=1))
                    all_scores.append(rois_probs)

            proposal_boxes = tf.concat(all_boxes, axis=0)  # (#lvl x BS) x K x 5
            proposal_boxes = tf.reshape(proposal_boxes, [-1, 5])  # (#lvl x BS x K) x 5

            proposal_scores = tf.concat(all_scores, axis=0)  # (#lvl x BS) x K
            proposal_scores = tf.reshape(proposal_scores, [-1])  # (#lvl x BS x 5) vector

            topk = tf.minimum(tf.size(proposal_scores), fpn_nms_topk)
            topk_scores, topk_indices = tf.nn.top_k(proposal_scores, k=topk, sorted=False)

            boxes_list.append(tf.gather(proposal_boxes, topk_indices))
            scores_list.append(tf.gather(proposal_scores, topk_indices))

        #
        #        boxes_list = []
        #        scores_list = []
        #
        #        for i in range(batch_size):
        #            batch_ind = tf.squeeze(tf.where(tf.equal(proposal_boxes[:, 0], i)), axis=1)
        #            image_scores = tf.gather(proposal_scores, batch_ind)
        #            image_boxes = tf.gather(proposal_boxes, batch_ind)
        #
        #            image_proposal_topk = tf.minimum(tf.size(image_scores), fpn_nms_topk//batch_size)
        #            image_proposal_scores, image_topk_indices = tf.nn.top_k(image_scores, k=image_proposal_topk, sorted=False)
        #            boxes_list.append(tf.gather(image_boxes, image_topk_indices))
        #            scores_list.append(image_proposal_scores)

        boxes = tf.concat(boxes_list, axis=0)
        scores = tf.concat(scores_list, axis=0)

        #        proposal_topk = tf.minimum(tf.size(proposal_scores), fpn_nms_topk)
    #        proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=False)
    #        proposal_boxes = tf.gather(proposal_boxes, topk_indices)

    else:
        raise RuntimeError("Only level-wise predictions are supported with batches")

    return tf.stop_gradient(boxes, name='boxes'), \
        tf.stop_gradient(scores, name='scores')
