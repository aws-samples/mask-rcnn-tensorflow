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
      boxes: nx4 floatbox

    Returns:
      n
    """
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


@under_name_scope()
def pairwise_intersection(boxlist1, boxlist2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=1)
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


@under_name_scope()
def pairwise_iou(boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4

    Returns:
      a tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = pairwise_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))



@under_name_scope()
def pairwise_iou_batch(proposal_boxes, gt_boxes, orig_gt_counts, batch_size):
    """Computes pairwise intersection-over-union between box collections.
    Args:
      proposal_boxes: K x 5  (batch_index, x1, y1, x2, y2)
      gt_boxes: BS x MaxNumGTs x 4
      orig_gt_counts: BS
    Returns:
        list of length BS, each element is output of pairwise_iou: N x M
        (where N is number of boxes for image and M is number of GTs for image)
    """

    prefix = "pairwise_iou_batch"

    # For each image index, extract a ?x4 boxlist and gt_boxlist

    per_images_iou = []
    for batch_idx in range(batch_size):

        box_mask_for_image = tf.equal(proposal_boxes[:, 0], batch_idx)

        single_image_boxes = tf.boolean_mask(proposal_boxes, box_mask_for_image)
        single_image_boxes = single_image_boxes[:, 1:]
        single_image_gt_boxes = gt_boxes[batch_idx, 0:orig_gt_counts[batch_idx], :]
        single_image_iou = pairwise_iou(single_image_boxes, single_image_gt_boxes)

        per_images_iou.append(single_image_iou)

    return per_images_iou
