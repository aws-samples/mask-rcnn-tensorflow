# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorpack.models import Conv2D, Conv2DTranspose, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary

from model.backbone import GroupNorm
from config import config as cfg
from utils.mixed_precision import mixed_precision_scope

@under_name_scope()
def maskrcnn_loss(mask_logits, fg_labels, fg_target_masks):
    """
    Args:
        mask_logits: Num_fg_boxes x num_category x H_roi x W_roi
        fg_labels: 1-D Num_fg_boxes, in 1~#class, int64
        fg_target_masks: Num_fg_boxes x H_roi x W_roi, float32
    Returns: mask loss
    """
    num_fg = tf.size(fg_labels, out_type=tf.int64) # scalar Num_fg_boxes
    indices = tf.stack([tf.range(num_fg), fg_labels - 1], axis=1)  # Num_fg_boxes x 2
    mask_logits = tf.gather_nd(mask_logits, indices)  # Num_fg_boxes x H_roi x W_roi
    mask_probs = tf.sigmoid(mask_logits)

    # add some training visualizations to tensorboard
    with tf.name_scope('mask_viz'):
        viz = tf.concat([fg_target_masks, mask_probs], axis=1)
        viz = tf.expand_dims(viz, 3)
        viz = tf.cast(viz * 255, tf.uint8, name='viz')
        tf.summary.image('mask_truth|pred', viz, max_outputs=10)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=fg_target_masks, logits=mask_logits)
    loss = tf.reduce_mean(loss, name='maskrcnn_loss')

    # Calculate the accuracy
    pred_label = mask_probs > 0.5
    truth_label = fg_target_masks > 0.5
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(pred_label, truth_label), tf.float32),
        name='accuracy')
    pos_accuracy = tf.logical_and(
        tf.equal(pred_label, truth_label),
        tf.equal(truth_label, True))
    pos_accuracy = tf.reduce_mean(tf.cast(pos_accuracy, tf.float32), name='pos_accuracy')
    fg_pixel_ratio = tf.reduce_mean(tf.cast(truth_label, tf.float32), name='fg_pixel_ratio')

    add_moving_summary(loss, accuracy, fg_pixel_ratio, pos_accuracy)
    return loss


@layer_register(log_shape=True)
def maskrcnn_upXconv_head(feature, num_category, seed_gen, num_convs, norm=None, fp16=False):
    """
    Args:
        feature: roi feature maps, Num_boxes x NumChannel x H_roi x W_roi,
        num_category(int): Number of total classes
        num_convs (int): number of convolution layers
        norm (str or None): either None or 'GN'

    Returns:
        mask_logits: Num_boxes x num_category x (2 * H_roi) x (2 * W_roi)
    """
    assert norm in [None, 'GN'], norm
    l = feature
    if fp16:
        l = tf.cast(l, tf.float16)
    with mixed_precision_scope(mixed=fp16):
      with argscope([Conv2D, Conv2DTranspose], data_format='channels_first',
                  kernel_initializer=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_out', seed=seed_gen.next(),
                      distribution='untruncated_normal' if get_tf_version_tuple() >= (1, 12) else 'normal')):
        # c2's MSRAFill is fan_out
        for k in range(num_convs):
            l = Conv2D('fcn{}'.format(k), l, cfg.MRCNN.HEAD_DIM, 3, activation=tf.nn.relu, seed=seed_gen.next())
            if norm is not None:
                if fp16: l = tf.cast(l, tf.float32)
                l = GroupNorm('gn{}'.format(k), l)
                if fp16: l = tf.cast(l, tf.float16)
        l = Conv2DTranspose('deconv', l, cfg.MRCNN.HEAD_DIM, 2, strides=2, activation=tf.nn.relu, seed=seed_gen.next()) # 2x upsampling
        l = Conv2D('conv', l, num_category, 1, seed=seed_gen.next())
    if fp16:
        l = tf.cast(l, tf.float32)
    return l

# Without Group Norm
def maskrcnn_up4conv_head(*args, **kwargs):
    return maskrcnn_upXconv_head(*args, num_convs=4, **kwargs)

# With Group Norm
def maskrcnn_up4conv_gn_head(*args, **kwargs):
    return maskrcnn_upXconv_head(*args, num_convs=4, norm='GN', **kwargs)
