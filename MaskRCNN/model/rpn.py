# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from turtle import shape
import tensorflow as tf
from MaskRCNN.performance import print_runtime_shape, print_runtime_tensor

from tensorpack import get_current_tower_context
from tensorpack.models import Conv2D, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope, under_name_scope

from config import config as cfg
from model_box import permute_boxes_coords


@layer_register(log_shape=True)
@auto_reuse_variable_scope
def rpn_head(featuremap, channel, num_anchors, seed_gen):
    """
    The RPN head that takes the feature map from the FPN and outputs bounding box logits.
    For every pixel on the feature maps, there are a certain number of anchors.
    The output will be:
    label logits: indicate whether there is an object for a certain anchor in one pixel
    box logits: The encoded box logits from fast-rcnn paper https://arxiv.org/abs/1506.01497
                page 5, in order to be consistent with the ground truth encoded boxes

    Args:
        featuremap: feature map for a single FPN layer, i.e. one from P23456
        channel: NumChannel of the feature map, scalar, default 256
        num_anchors(NA): # of anchors for each pixel in the current feature map, scalar, default 3
    Returns:
        label_preds: BS x H_feature x W_feature x NA
        box_preds: BS x H_feature x W_feature X (NA * 4)
    """
    
    with argscope(Conv2D, data_format='channels_first' if cfg.TRAIN.RPN_NCHW else 'channels_last',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=seed_gen.next())):
        hidden = Conv2D('conv0', featuremap, channel, 3, activation=tf.nn.relu, seed=seed_gen.next())
        label_preds = Conv2D('class', hidden, num_anchors, 1, seed=seed_gen.next())
        box_preds = Conv2D('box', hidden, 4 * num_anchors, 1, seed=seed_gen.next())
       
        if cfg.TRAIN.RPN_NCHW:
            label_preds = tf.transpose(label_preds, [0, 2, 3, 1])  # BS x H_feature x W_feature x NA
            box_preds = tf.transpose(box_preds, [0, 2, 3, 1])  # BS x  H_feature x W_feature X (NA*4)
        
    return label_preds, box_preds


@under_name_scope()
def top_k_boxes( scores, k, boxes):
    """A wrapper that returns top-k scores and corresponding boxes.

    This functions selects the top-k scores and boxes as follows.

    indices = argsort(scores)[:k]
    scores = scores[indices]
    

    Args:
        scores: a tensor with a shape of [batch_size, N]. N is the number of scores.
        k: an integer for selecting the top-k elements.
        boxes:  [batch_size, N, 4].
    Returns:
        rois: [batch_size, N, 4]
    """
    batch_size = tf.shape(scores)[0]
    _, top_k_indices = tf.math.top_k(scores, k=k)
    
    boxes_index_offsets = tf.range(batch_size) * tf.shape(boxes)[1]
    boxes_indices = tf.reshape(top_k_indices + tf.expand_dims(boxes_index_offsets, 1), [-1])
    boxes = tf.reshape(tf.gather(tf.reshape(boxes, [-1, 4]), boxes_indices), [batch_size, -1, 4])

    return boxes

@under_name_scope()
def rpn_losses(labels_gt, boxes_gt, labels_pred, boxes_pred):
    """
    Calculate the rpn loss for one FPN layer for a single image.
    The ground truth(GT) anchor labels and anchor boxes has been preprocessed to fit
    the dimensions of FPN feature map. The GT boxes are encoded from fast-rcnn paper
    https://arxiv.org/abs/1506.01497 page 5.

    Args:
        labels_gt: GT anchor labels, H_feature x W_feature x NA
        boxes_gt: GT boxes for each anchor, H_feature x W_feature x NA x 4, encoded
        labels_pred: label predictions from the rpn head, H_feature x W_feature x NA
        boxes_pred: box predictions from the rpn head, H_feature x W_feature x NA x 4
    Returns:
        label_loss, box_loss
    """
    
    #labels_gt = print_runtime_shape("labels_gt", labels_gt)
    #boxes_gt = print_runtime_shape("boxes_gt", boxes_gt)
    #labels_pred = print_runtime_shape("labels_pred", labels_pred)
    #boxes_pred = print_runtime_shape("boxes_pred", boxes_pred)

    valid_mask = tf.stop_gradient(tf.math.not_equal(labels_gt, -1))
    pos_mask = tf.stop_gradient(tf.math.equal(labels_gt, 1))
    num_valid = tf.stop_gradient(tf.math.count_nonzero(valid_mask, dtype=tf.int32), name='num_valid_anchor')
    num_pos =  tf.identity(tf.math.count_nonzero(pos_mask, dtype=tf.int32), name='num_pos_anchor')
    # In FPN. even num_valid could be 0.

    valid_gt_labels = tf.boolean_mask(labels_gt, valid_mask)
    valid_label_preds = tf.boolean_mask(labels_pred, valid_mask)

    # Per-level loss summaries in FPN may appear lower due to the use of a small placeholder.
    # But the total RPN loss will be fine.  TODO make the summary op smarter

    #num_valid = print_runtime_tensor("num_valid", num_valid)
    #num_pos = print_runtime_tensor("num_pos", num_pos)

    #valid_gt_labels = print_runtime_tensor("valid_gt_labels", valid_gt_labels)
    #valid_label_preds = print_runtime_tensor("valid_label_preds", valid_label_preds)

    zero_loss = 0.0
    label_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(valid_gt_labels, tf.float32), logits=valid_label_preds)
    label_loss = tf.reduce_sum(label_loss) * (1. / cfg.RPN.BATCH_PER_IM)
    label_loss = tf.where(tf.equal(num_valid, 0), zero_loss, label_loss, name='label_loss')

    pos_gt_boxes = tf.boolean_mask(boxes_gt, pos_mask)
    pos_box_preds = tf.boolean_mask(boxes_pred, pos_mask)

    #pos_gt_boxes = print_runtime_tensor("pos_gt_boxes", pos_gt_boxes)
    #pos_box_preds = print_runtime_tensor("pos_box_preds", pos_box_preds)

    delta = 1.0 / 9
    box_loss = tf.compat.v1.losses.huber_loss(pos_gt_boxes, pos_box_preds, delta=delta, reduction="none")
    box_loss = tf.reduce_sum(box_loss) / delta
    box_loss = box_loss * (1. / cfg.RPN.BATCH_PER_IM)

    #num_pos = print_runtime_tensor("num_pos", num_pos)
    box_loss = tf.where(tf.equal(num_pos, 0), zero_loss, box_loss, name='box_loss')

    #box_loss = print_runtime_tensor("box_loss", box_loss)
    #label_loss = print_runtime_tensor("label_loss", label_loss)

    # add_moving_summary(label_loss, box_loss, nr_valid, nr_pos)
    return [label_loss, box_loss]

@under_name_scope(name_scope="batch_rpn_losses")
def batch_rpn_losses(multilevel_rpn_gt, multilevel_label_preds, multilevel_box_preds, orig_image_shape2d):
    """
    Calculates the rpn loss for all FPN layers for a batch of images.

    Args:
        multilevel_rpn_gt: #lvl RPNGroundTruth
        multilevel_label_preds: [BS X H_feature x W_feature x NA] * Num_levels
        multilevel_box_preds: [BS X H_feature x W_feature x NA x 4] * Num_levels
        orig_image_shape2d: BS X 2 (orig_image_h, orig_image_w)
    Returns:
        label_loss, box_loss
    """
    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_rpn_gt) == num_lvl
    assert len(multilevel_label_preds) == num_lvl
    assert len(multilevel_box_preds) == num_lvl

    losses = []

    mult = float (cfg.FPN.RESOLUTION_REQUIREMENT)  # the image is padded so that it is a multiple of this (32 with default config).
    orig_image_hw_after_fpn_padding = tf.math.ceil(tf.cast(orig_image_shape2d, tf.float32) / mult) * mult
    featuremap_dims_per_level = []
    for lvl, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
        featuremap_dims_float = orig_image_hw_after_fpn_padding / float(stride)
        featuremap_dims_per_level.append(tf.cast(tf.math.floor(featuremap_dims_float + 0.5), tf.int32))  # Fix bankers rounding
    
    for lvl in range(num_lvl):
        rpn_gt = multilevel_rpn_gt[lvl]

        lvl_labels_gt = rpn_gt.gt_labels
        lvl_encoded_boxes_gt = rpn_gt.encoded_gt_boxes()
        lvl_labels_pred = multilevel_label_preds[lvl]
        lvl_boxes_pred = multilevel_box_preds[lvl]

        lvl_featuremap_dims = featuremap_dims_per_level[lvl]

        for i in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
            image_lvl_rpn_labels_gt = lvl_labels_gt[i]
            image_lvl_rpn_boxes_gt = lvl_encoded_boxes_gt[i]
            image_lvl_labels_pred = lvl_labels_pred[i]
            image_lvl_boxes_pred = lvl_boxes_pred[i]

            image_lvl_dims = lvl_featuremap_dims[i]

            image_lvl_rpn_labels_gt_narrowed = image_lvl_rpn_labels_gt[:image_lvl_dims[0], :image_lvl_dims[1] ,:] 
            image_lvl_rpn_boxes_gt_narrowed = image_lvl_rpn_boxes_gt[:image_lvl_dims[0], :image_lvl_dims[1] ,: ,:]
            image_lvl_labels_pred_narrowed = image_lvl_labels_pred[:image_lvl_dims[0], :image_lvl_dims[1] ,:] 
            image_lvl_boxes_pred_narrowed = image_lvl_boxes_pred[:image_lvl_dims[0], :image_lvl_dims[1] ,: ,:] 
    
            image_lvl_label_loss, image_lvl_box_loss = rpn_losses(
                image_lvl_rpn_labels_gt_narrowed, 
                image_lvl_rpn_boxes_gt_narrowed,
                image_lvl_labels_pred_narrowed, 
                image_lvl_boxes_pred_narrowed,
                name_scope=f'level{lvl+2}')
            losses.extend([image_lvl_label_loss, image_lvl_box_loss])

    total_label_loss = tf.add_n(losses[::2])
    total_box_loss = tf.add_n(losses[1::2])

    return [total_label_loss, total_box_loss]

@under_name_scope()
def generate_fpn_proposals(image_shape2d,
                           all_anchors_fpn,
                           multilevel_label_preds,
                           multilevel_bbox_preds,
                           orig_image_shape2d,
                           batch_size):
    """
    Generating the rois from the box logits and pick K with top label scores as
    the box proposals.

    Args:
        image_shape2d: image shape H,W  2
        all_anchors_fpn:  #lvl [ H_feature x W_feature x NA X 4] 
        multilevel_label_preds:  #lvl [ BS x H_feature x W_feature x NA ]  
        multilevel_bbox_preds:   #lvl [ BS  x H_feature x W_feature X (NA * 4)] 
        orig_image_shape2d: Original (prepadding) image dimensions (h,w)   BS x 2
        batch_size: Batch size
    Returns:
        boxes: BS X Top_K X 4 
    """
    
    training = get_current_tower_context().is_training
    num_lvl = len(cfg.FPN.ANCHOR_STRIDES)
    assert len(multilevel_label_preds) == num_lvl
    assert len(multilevel_bbox_preds) == num_lvl

    if cfg.FPN.PROPOSAL_MODE == 'Level':
        fpn_pre_nms_topk = cfg.RPN.TRAIN_PER_LEVEL_PRE_NMS_TOPK if training else cfg.RPN.TEST_PER_LEVEL_PRE_NMS_TOPK
        fpn_post_nms_topk = cfg.RPN.TRAIN_PER_LEVEL_POST_NMS_TOPK if training else cfg.RPN.TEST_PER_LEVEL_POST_NMS_TOPK

        orig_image_shape2d = tf.cast(orig_image_shape2d, tf.float32)
        shape2d = tf.cast(image_shape2d, tf.float32)
        shape2d = tf.expand_dims(shape2d, 0)
        shape2d = tf.tile(shape2d, [batch_size, 1])
        
        rois = []
        scores = []

        for lvl in range(num_lvl):
            with tf.name_scope(f'Lvl{lvl}'):
                spatial_scale=tf.constant([1.0 / cfg.FPN.ANCHOR_STRIDES[lvl]], dtype=tf.float32)
                spatial_scale = tf.expand_dims(spatial_scale, 0)
                spatial_scale = tf.tile(spatial_scale, [batch_size, 1])
                
                im_info = tf.concat([shape2d, spatial_scale, orig_image_shape2d], axis=-1)

                singlelevel_anchor_boxes = all_anchors_fpn[lvl] # H_feature x W_feature x NA x 4 (x1, y1, x2, y2)
                level_shape = tf.shape(singlelevel_anchor_boxes)
                singlelevel_anchor_boxes = permute_boxes_coords(singlelevel_anchor_boxes) # H_feature x W_feature x NA x 4 (y1, x1, y2, x2)
                singlelevel_anchor_boxes = tf.reshape(singlelevel_anchor_boxes, [level_shape[0], level_shape[1], -1] ) # H_feature x W_feature x NA * 4 
              
                singlelevel_scores = multilevel_label_preds[lvl] # BS x H_feature x W_feature x NA
                singlelevel_scores = tf.math.sigmoid(singlelevel_scores) 
                
                singlelevel_bbox_deltas = multilevel_bbox_preds[lvl] # BS  x H_feature x W_feature X (NA * 4)
                sbd_shape = tf.shape(singlelevel_bbox_deltas)
                singlelevel_bbox_deltas = tf.reshape(singlelevel_bbox_deltas, [sbd_shape[0], sbd_shape[1], sbd_shape[2], -1, 4] ) # BS  x H_feature x W_feature X NA X 4 (dx, dy, dw, dh)
                singlelevel_bbox_deltas = permute_boxes_coords(singlelevel_bbox_deltas) # BS  x H_feature x W_feature X NA X 4 (dy, dx, dh, dw)
                singlelevel_bbox_deltas = tf.reshape(singlelevel_bbox_deltas, [sbd_shape[0], sbd_shape[1], sbd_shape[2], -1] ) # BS  x H_feature x W_feature X NA * 4 

                singlelevel_anchor_boxes_narrowed = singlelevel_anchor_boxes[:sbd_shape[1], :sbd_shape[2], :]

                lvl_rois, lvl_rois_scores = tf.image.generate_bounding_box_proposals (
                                        scores=singlelevel_scores,
                                        bbox_deltas=singlelevel_bbox_deltas,
                                        image_info=im_info,
                                        anchors=singlelevel_anchor_boxes_narrowed,
                                        pre_nms_topn=fpn_pre_nms_topk,
                                        post_nms_topn=fpn_post_nms_topk,
                                        nms_threshold=cfg.RPN.PROPOSAL_NMS_THRESH,
                                        min_size=cfg.RPN.MIN_SIZE)

                lvl_rois = permute_boxes_coords(lvl_rois) #  BS  x Top_k X 4 (x1, y1, x2, y2)
                rois.append(lvl_rois)
                scores.append(lvl_rois_scores)

        scores = tf.concat(scores, axis=1)
        rois = tf.concat(rois, axis=1)

        with tf.name_scope('post_nms_topk'):
            # Selects the top-k rois
            image_post_nms_topk = cfg.RPN.TRAIN_IMAGE_POST_NMS_TOPK if training else cfg.RPN.TEST_IMAGE_POST_NMS_TOPK
            image_post_nms_topk = tf.minimum(tf.shape(rois)[1], image_post_nms_topk)
            top_k_rois = top_k_boxes( scores, k=image_post_nms_topk, boxes=rois)
    else:
        raise ValueError("Only FPN.PROPOSAL_MODE == 'Level' config is supported")

    return tf.stop_gradient(top_k_rois, name='boxes')
