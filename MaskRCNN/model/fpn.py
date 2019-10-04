# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import tensorflow as tf

from tensorpack.models import Conv2D, FixedUnPooling, MaxPooling, layer_register
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary

from model.backbone import GroupNorm
from config import config as cfg
from utils.box_ops import area as tf_area
from utils.mixed_precision import mixed_precision_scope

@layer_register(log_shape=True)
def fpn_model(features, seed_gen, fp16=False):
    """
    Args:
        features ([tf.Tensor]): ResNet features c2-c5

    Returns:
        [tf.Tensor]: FPN features p2-p6
    """
    assert len(features) == 4, features
    num_channel = cfg.FPN.NUM_CHANNEL

    use_gn = cfg.FPN.NORM == 'GN'

    def upsample2x(name, x):
        dtype_str = 'float16' if fp16 else 'float32'
        return FixedUnPooling(
            name, x, 2, unpool_mat=np.ones((2, 2), dtype=dtype_str),
            data_format='channels_first')

        # tf.image.resize is, again, not aligned.
        # with tf.name_scope(name):
        #     shape2d = tf.shape(x)[2:]
        #     x = tf.transpose(x, [0, 2, 3, 1])
        #     x = tf.image.resize_nearest_neighbor(x, shape2d * 2, align_corners=True)
        #     x = tf.transpose(x, [0, 3, 1, 2])
        #     return x

    with mixed_precision_scope(mixed=fp16):
        with argscope(Conv2D, data_format='channels_first',
                  activation=tf.identity, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(scale=1., seed=seed_gen.next())): #kernel_initializer=tf.glorot_normal_initializer(seed=1234)):
            lat_2345 = [Conv2D('lateral_1x1_c{}'.format(i + 2), c, num_channel, 1, seed=seed_gen.next())
                        for i, c in enumerate(features)]
            #lat_2345 = [tf.keras.layers.Conv2D(filters=num_channel, activation=tf.identity,
            #                                   use_bias=True, data_format='channels_first',
            #                                   kernel_initializer=tf.keras.initializers.glorot_normal(seed=1234),
            #                                   kernel_size=1)(c) for i,c in enumerate(features)]
            if use_gn:
                if fp16:
                    lat_2345 = [tf.cast(i, tf.float32) for i in lat_2345]
                lat_2345 = [GroupNorm('gn_c{}'.format(i + 2), c) for i, c in enumerate(lat_2345)]
                if fp16:
                    lat_2345 = [tf.cast(i, tf.float16) for i in lat_2345]
            lat_sum_5432 = []
            for idx, lat in enumerate(lat_2345[::-1]):
                if idx == 0:
                    lat_sum_5432.append(lat)
                else:
                    lat = lat + upsample2x('upsample_lat{}'.format(6 - idx), lat_sum_5432[-1])
                    #lat = lat + tf.keras.layers.UpSampling2D(data_format='channels_first')(lat_sum_5432[-1])
                    lat_sum_5432.append(lat)
            p2345 = [Conv2D('posthoc_3x3_p{}'.format(i + 2), c, num_channel, 3, seed=seed_gen.next())
                     for i, c in enumerate(lat_sum_5432[::-1])]
            if use_gn:
                if fp16:
                    p2345 = [tf.cast(i, tf.float32) for i in p2345]
                p2345 = [GroupNorm('gn_p{}'.format(i + 2), c) for i, c in enumerate(p2345)]
                if fp16:
                    p2345 = [tf.cast(i, tf.float16) for i in p2345]
            #p6 = MaxPooling('maxpool_p6', p2345[-1], pool_size=1, strides=2, data_format='channels_first', padding='VALID')
            p6 = tf.keras.layers.MaxPool2D(pool_size=1, strides=2, data_format='channels_first')(p2345[-1])
            if fp16:
                return [tf.cast(l, tf.float32) for l in p2345] + [tf.cast(p6, tf.float32)]

            return p2345 + [p6]

@under_name_scope()
def fpn_map_rois_to_levels(boxes):
    """
    Assign boxes to level 2~5.

    Args:
        boxes: t x 5, t is the number of sampled boxes

    Returns:
        level_ids[tf.Tensor]: 4 tensors for level 2-5. Each tensor is a vector of indices of boxes in its level.
        level_boxes[tf.Tensor]: 4 tensors, the gathered boxes in each level.

    Be careful that the returned tensor could be empty.
    """
    sqrtarea = tf.sqrt(tf_area(boxes[:,1:]))
    # Map equation from the FPN paper: https://arxiv.org/abs/1612.03144, page 4
    # k = [k0 + log2(sqrt(wh)/224)]
    level = tf.cast(tf.floor(
        4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))), tf.int32)

    # RoI levels range from 2~5 (not 6)
    level_ids = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),   # == is not supported
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5)]
    level_ids = [tf.reshape(x, [-1], name='roi_level{}_id'.format(i + 2))
                 for i, x in enumerate(level_ids)]
    num_in_levels = [tf.size(x, name='num_roi_level{}'.format(i + 2))
                     for i, x in enumerate(level_ids)]
    add_moving_summary(*num_in_levels)

    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    return level_ids, level_boxes

@under_name_scope()
def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.
    Args:
        image: NCHW
        boxes: nx4, x1y1x2y2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
        boxes = boxes + 1

    @under_name_scope()
    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.
        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
         (0.0, 0.0) is the same as pixel value (0, 0))
        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize
        Returns:
            y1x1y2x2
        """
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    image_shape = tf.shape(image)[2:]

    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])   # nhwc
    ret = tf.image.crop_and_resize(
        image, boxes, tf.cast(box_ind, tf.int32),
        crop_size=[crop_size, crop_size])
    ret = tf.transpose(ret, [0, 3, 1, 2])   # ncss
    return ret

@under_name_scope()
def roi_align(featuremap, boxes, resolution):
    """
    Args:
        featuremap: 1xCxHxW
        boxes: Nx4 floatbox
        resolution: output spatial resolution
    Returns:
        NxCx res x res
    """
    # sample 4 locations per roi bin
    ret = tf.image.crop_and_resize(
        featuremap, boxes,
        tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
        resolution * 2)
    try:
        avgpool = tf.nn.avg_pool2d
    except AttributeError:
        avgpool = tf.nn.avg_pool
    ret = avgpool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return ret

@under_name_scope()
def multilevel_roi_align(features, rcnn_boxes, resolution):
    """
    Args:
        features ([tf.Tensor]): 4 FPN feature level P2-5, each with BS X NumChannel X H_feature X W_feature
        rcnn_boxes (tf.Tensor): t x 5, t is the number of sampled boxes
        resolution (int): output spatial resolution, scalar
    Returns:
        all_rois: Num_fg_boxes x NumChannel x H_roi x W_roi
    """
    assert len(features) == 4, features
    # Reassign rcnn_boxes to levels
    level_ids, level_boxes = fpn_map_rois_to_levels(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
        with tf.name_scope('roi_level{}'.format(i + 2)):

            # coordinate system fix for boxes
            boxes = tf.concat((boxes[:,:1], boxes[:,1:] - 0.5*cfg.FPN.ANCHOR_STRIDES[i]), axis=1)

            # This is a custom tensorflow op for doing ROI align. See CODEBASE.md for more info
            #roi_feature_maps = tf.roi_align(featuremap,
            #                                boxes,
            #                                pooled_height=resolution,
            #                                pooled_width=resolution,
            #                                spatial_scale=1.0 / cfg.FPN.ANCHOR_STRIDES[i],
            #                                sampling_ratio=2)
            roi_feature_maps = roi_align(featuremap, boxes, resolution)
            all_rois.append(roi_feature_maps)

    # this can fail if using TF<=1.8 with MKL build
    all_rois = tf.concat(all_rois, axis=0)  # NCHW
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm)
    return all_rois
