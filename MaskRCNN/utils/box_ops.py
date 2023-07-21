# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
# File: box_ops.py

import tensorflow as tf

from tensorpack.tfutils.scope_utils import under_name_scope


"""
This file is modified from
https://github.com/tensorflow/models/blob/master/object_detection/core/box_list_ops.py
"""


@under_name_scope()
def area(boxes):
    """
    Args:
      boxes: BS  X N X 4 (x1, y1, x2, y2)

    Returns:
      BS X N
    """
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=-1)
    return (y_max - y_min) * (x_max - x_min)


@under_name_scope()
def pairwise_intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxes1: BS X N X 4 (x1, y1, x2, y2) 
      boxes2: BS X M X 4 (x1, y1, x2, y2)

    Returns:
      a tensor with shape [BS X N X M] representing pairwise intersections
    """
    x_min1, y_min1, x_max1, y_max1  = tf.split(boxes1, 4, axis=-1)
    x_min2, y_min2, x_max2, y_max2  = tf.split(boxes2, 4, axis=-1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2, perm=[0, 2, 1]))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2, perm=[0, 2, 1]))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2, perm=[0, 2, 1]))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2, perm=[0, 2, 1]))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


@under_name_scope()
def pairwise_iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxes1: BS X N X 4 (x1, y1, x2, y2)
      boxes2: BS X M X 4 (x1, y1, x2, y2)

    Returns:
      a tensor with shape [BS, N, M] representing pairwise iou scores.
    """
    intersections = pairwise_intersection(boxes1, boxes2)
    areas1 = area(boxes1)
    areas2 = area(boxes2)
    unions = (areas1 + tf.transpose(areas2, [0, 2, 1]) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))