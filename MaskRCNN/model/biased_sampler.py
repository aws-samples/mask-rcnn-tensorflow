# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
from MaskRCNN.performance import print_runtime_shape, print_runtime_tensor

from config import config as cfg
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from utils.box_ops import pairwise_iou


@under_name_scope()
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels, orig_gt_counts, seed_gen):
    """
    Sample boxes according to the predefined fg(foreground) boxes and bg(background) boxes ratio
    #fg(foreground) is guaranteed to be > 0, because ground truth boxes will be added as proposals.
    Args:
        boxes: BS X N X 4 (x1, y1, x2, y2)
        gt_boxes: Groundtruth boxes, BS x MaxGT x 4(x1, y1, x2, y2)
        gt_labels: BS x MaxGT, int32
        orig_gt_counts: BS # The number of ground truths in the data. Use to unpad gt_labels and gt_boxes
    Returns:
        sampled_boxes: BS X Nsamples X 4(x1, y1, x2, y2) , the rois
        sampled_labels: BS X Nsamples int64 labels, in [0, #class). Positive means foreground.
        fg_inds_wrt_gt: #fg indices, each in range [0, m-1]. It contains the matching GT index of each foreground roi.
    """

    ious = pairwise_iou(boxes, gt_boxes)
    proposal_metrics_batch(ious, cfg.TRAIN.BATCH_SIZE_PER_GPU, orig_gt_counts)
    
    def sample_fg_bg(ious, num_samples):
        """
        Sample rows from the iou so that:
            - you have the correct ratio of fg/bg,
            - The total number of sampled rows matches FRCNN.BATCH_PER_IM 
        FG/BG is determined based on whether the proposal has an IOU with a GT that crosses the FG_THRESH
        Args:
            ious: N x M
            num_samples: Num of samples, int32
        Returns:
            num_fg: number of foreground samples
            num_bg: numberof background samples
            fg_inds: foreground samples indicies
            bg_inds: background samples indices
        """
        fg_mask = tf.reduce_max(ious, axis=-1) >= cfg.FRCNN.FG_THRESH # N vector
        fg_inds = tf.where(fg_mask)
        image_num_fg = tf.size(fg_inds)
        bg_inds = tf.where(tf.math.logical_not(fg_mask))
        image_num_bg = tf.size(bg_inds)

        max_fg_num = tf.cast(tf.math.multiply(tf.cast(num_samples, dtype=tf.float32), cfg.FRCNN.FG_RATIO), dtype=tf.int32)
        min_bg_num = tf.cast(num_samples - max_fg_num, dtype=tf.int32)

        num_fg = tf.cond(tf.math.less_equal(image_num_fg, max_fg_num), 
                lambda: image_num_fg, 
                lambda: tf.cond(tf.math.greater_equal(image_num_bg, min_bg_num), 
                    lambda: max_fg_num,  
                    lambda: num_samples - image_num_bg ))
     
        num_bg = num_samples  - num_fg

        fg_inds = tf.reshape(tf.random.shuffle(fg_inds, seed=seed_gen.next('sampler'))[0:num_fg, :], [-1])
        bg_inds = tf.reshape(tf.random.shuffle(bg_inds, seed=seed_gen.next('sampler'))[0:num_bg, :], [-1])

        return num_fg, num_bg, fg_inds, bg_inds


    sampled_boxes = []
    sampled_labels_gt = []
    sampled_boxes_gt = []
    sampled_fg_gt_indices = []
    num_bgs = []
    num_fgs = []
    num_boxes = tf.shape(boxes)[1]
    num_samples = tf.cond(tf.math.less_equal(cfg.FRCNN.BATCH_PER_IM, num_boxes), 
                lambda: cfg.FRCNN.BATCH_PER_IM, lambda: num_boxes)

    for i in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
        gt_count = orig_gt_counts[i]
        image_gt_boxes = gt_boxes[i, 0:gt_count, :]

        image_boxes = boxes[i] # N X 4 (x1, y1, x2, y2)
        image_boxes = tf.concat([image_boxes, image_gt_boxes], axis=0)
        
        image_ious = ious[i]
        image_ious = image_ious[:, 0:gt_count]
        
        image_ious = tf.concat([image_ious, tf.eye(gt_count)], axis=0)

        num_fg, num_bg, fg_inds, bg_inds = sample_fg_bg(image_ious, num_samples)

        #fg_inds = print_runtime_tensor("fg_inds", fg_inds)

        num_fgs.append(num_fg)
        num_bgs.append(num_bg)
        image_inds = tf.concat([fg_inds, bg_inds], axis=0) # indices w.r.t all n proposal boxes

        image_sampled_boxes = tf.gather(image_boxes, image_inds)  # ? X 4 (x1, y1, x2, y2)
        sampled_boxes.append(image_sampled_boxes)

        image_gt_labels = gt_labels[i, 0:gt_count]   # gt_count

        best_iou_ind = tf.argmax(image_ious, axis=1)
        image_fg_gt_indices = tf.gather(best_iou_ind, fg_inds)  # num_fg
        image_fg_gt_labels = tf.gather(image_gt_labels, image_fg_gt_indices)
        sampled_image_gt_labels = tf.concat([image_fg_gt_labels, tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0)
        sampled_labels_gt.append(sampled_image_gt_labels)

        image_gt_boxes_fg = tf.gather(image_gt_boxes, image_fg_gt_indices)
        image_gt_boxes_bg = tf.repeat(tf.expand_dims(tf.zeros_like(bg_inds, dtype=tf.float32), axis=1), 4, axis=1)
        sampled_image_gt_boxes = tf.concat([image_gt_boxes_fg, image_gt_boxes_bg], axis=0)
        sampled_boxes_gt.append(sampled_image_gt_boxes)
        sampled_fg_gt_indices.append(tf.stop_gradient(image_fg_gt_indices, name="image_fg_gt_ind"))

    total_num_fgs = tf.add_n(num_fgs, name="num_fg")
    total_num_bgs = tf.add_n(num_bgs, name="num_bg")

    #total_num_fgs = print_runtime_tensor("total_num_fgs", total_num_fgs)
    #total_num_bgs = print_runtime_tensor("total_num_bgs", total_num_bgs)

    add_moving_summary(total_num_fgs, total_num_bgs)

    sampled_boxes = tf.stack(sampled_boxes, axis=0)    # BS X FRCNN.BATCH_PER_IM x 4
    sampled_labels_gt = tf.stack(sampled_labels_gt, axis=0)  # BS X FRCNN.BATCH_PER_IM
    sampled_boxes_gt = tf.stack(sampled_boxes_gt, axis=0)

    # stop the gradient -- they are meant to be training targets
    sampled_boxes = tf.stop_gradient(sampled_boxes, name='sampled_boxes')
    sampled_labels_gt = tf.stop_gradient(sampled_labels_gt, name='sampled_gt_labels')
    sampled_boxes_gt = tf.stop_gradient(sampled_boxes_gt, name="sampled_gt_boxes")

    return sampled_boxes, sampled_labels_gt, sampled_boxes_gt, sampled_fg_gt_indices


@under_name_scope(name_scope="proposal_metrics")
def proposal_metrics_batch(ious, batch_size, orig_gt_counts):
    """
    Add summaries for RPN proposals.
    Args:
         ious: pairwise intersection-over-union between rpn boxes and gt boxes
                         BS X Num_rpn_boxes x Num_gt_boxes
    """
    # find best roi for each gt, for summary only
    thresholds = [0.3, 0.5]
    best_ious = []
    mean_best_ious = []
    recalls = {}
    for th in thresholds:
        recalls[th] = []

    for i in range(batch_size):
        image_iou = ious[i]
        image_orig_gt_count = orig_gt_counts[i]
        image_iou = image_iou[:, :image_orig_gt_count]

        best_iou = tf.math.reduce_max(image_iou, axis=0)
        #best_iou = print_runtime_tensor("best_iou", best_iou)
        best_ious.append(best_iou)

        mean_best_ious.append(tf.math.reduce_mean(best_iou))

        for th in thresholds:
            recall = tf.truediv(
                tf.math.count_nonzero(best_iou >= th),
                tf.size(best_iou, out_type=tf.int64))
            recalls[th].append(recall)

    all_mean_best_ious = tf.stack(mean_best_ious)
    mean_of_mean_best_iou = tf.math.reduce_mean(all_mean_best_ious, name='best_iou_per_gt')
    #mean_of_mean_best_iou = print_runtime_tensor("mean_of_mean_best_iou", mean_of_mean_best_iou)
    summaries = [mean_of_mean_best_iou]
    for th in thresholds:
        recall = tf.math.reduce_mean(tf.stack(recalls[th]), name='recall_iou{}'.format(th))
        summaries.append(recall)

    add_moving_summary(*summaries)


