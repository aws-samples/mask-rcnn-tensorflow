# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import tensorflow as tf

from config import config as cfg
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from utils.box_ops import pairwise_iou_batch


@under_name_scope()
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels, orig_gt_counts, batch_size, seed_gen):
    """
    Sample boxes according to the predefined fg(foreground) boxes and bg(background) boxes ratio
    #fg(foreground) is guaranteed to be > 0, because ground truth boxes will be added as proposals.
    Args:
        boxes: K x 5 region proposals. [batch_index, x1, y1, x2, y2]
        gt_boxes: Groundtruth boxes, BS x MaxGT x 4(x1, y1, x2, y2)
        gt_labels: BS x MaxGT, int32
        orig_gt_counts: BS # The number of ground truths in the data. Use to unpad gt_labels and gt_boxes
    Returns:
        sampled_boxes: tx5, the rois
        sampled_labels: t int64 labels, in [0, #class). Positive means foreground.
        fg_inds_wrt_gt: #fg indices, each in range [0, m-1]. It contains the matching GT of each foreground roi.
    """

    # per_image_ious: list of len BS [N x M] -- N is Num_rpn_boxes and M is Num_gt_boxes, for one image
    per_image_ious = pairwise_iou_batch(boxes, gt_boxes, orig_gt_counts, batch_size=batch_size)

    proposal_metrics_batch(per_image_ious)


    ious = []
    best_iou_inds = []
    for i in range(batch_size):
        image_ious = per_image_ious[i]
        gt_count = orig_gt_counts[i]

        single_image_gt_boxes = gt_boxes[i, :gt_count, :]

        single_image_gt_boxes = tf.pad(single_image_gt_boxes, [[0,0], [1,0]], mode="CONSTANT", constant_values=i)
        boxes = tf.concat([boxes, single_image_gt_boxes], axis=0)

        iou = tf.concat([image_ious, tf.eye(gt_count)], axis=0)  # (N+M) x M

        best_iou_ind = tf.argmax(iou, axis=1)   # A vector with the index of the GT with the highest IOU,
                                                # (length #proposals (N+M), values all in 0~m-1)

        ious.append(iou)
        best_iou_inds.append(best_iou_ind)

    def sample_fg_bg(iou):
        """
        Sample rows from the iou so that:
            - you have the correct ratio of fg/bg,
            - The total number of sampled rows matches FRCNN.BATCH_PER_IM (unless there are insufficient rows)
        FG/BG is determined based on whether the proposal has an IOU with a GT that crosses the FG_THRESH
        Args:
            iou: (N+M) x M
        Returns:
            fg_inds: List of rows indices (0:N+M) from iou that are fg and have iou at least FRCNN.FG_THRESH
            bg_inds: List of rows indices (0:N+M) from iou that are bg
        """
        fg_mask = tf.reduce_max(iou, axis=1) >= cfg.FRCNN.FG_THRESH # N+M vector

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        num_fg = tf.minimum(int(
            cfg.FRCNN.BATCH_PER_IM * cfg.FRCNN.FG_RATIO),
            tf.size(fg_inds))
        fg_inds = tf.random_shuffle(fg_inds, seed=seed_gen.next('sampler'))[:num_fg]
        # fg_inds = fg_inds[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            cfg.FRCNN.BATCH_PER_IM - num_fg,
            tf.size(bg_inds))
        bg_inds = tf.random_shuffle(bg_inds, seed=seed_gen.next('sampler'))[:num_bg]
        # bg_inds = bg_inds[:num_bg]


        return num_fg, num_bg, fg_inds, bg_inds


    all_ret_boxes = []
    all_ret_labels = []
    all_fg_inds_wrt_gt = []
    num_bgs = []
    num_fgs = []
    for i in range(batch_size):
        # ious[i] = print_runtime_tensor("ious[i]", ious[i], prefix=prefix)
        num_fg, num_bg, fg_inds, bg_inds = sample_fg_bg(ious[i])

        num_fgs.append(num_fg)
        num_bgs.append(num_bg)

        best_iou_ind = best_iou_inds[i]

        fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)  # num_fg

        all_indices = tf.concat([fg_inds, bg_inds], axis=0)  # indices w.r.t all n+m proposal boxes

        box_mask_for_image = tf.equal(boxes[:, 0], i) # Extract boxes for a single image so we can apply all_indices as mask

        single_images_row_indices = tf.squeeze(tf.where(box_mask_for_image), axis=1)
        single_image_boxes = tf.gather(boxes, single_images_row_indices) # ?x5
        single_image_ret_boxes = tf.gather(single_image_boxes, all_indices)  # ?x5
        all_ret_boxes.append(single_image_ret_boxes)

        gt_count = orig_gt_counts[i]
        single_image_gt_labels = gt_labels[i, 0:gt_count]   # Vector of length #gts

        single_image_ret_labels = tf.concat(
            [tf.gather(single_image_gt_labels, fg_inds_wrt_gt),
             tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0)

        all_ret_labels.append(single_image_ret_labels)
        all_fg_inds_wrt_gt.append(fg_inds_wrt_gt)

    total_num_fgs = tf.add_n(num_fgs, name="num_fg")
    total_num_bgs = tf.add_n(num_bgs, name="num_bg")
    add_moving_summary(total_num_fgs, total_num_bgs)

    ret_boxes = tf.concat(all_ret_boxes, axis=0)    # ? x 5
    ret_labels = tf.concat(all_ret_labels, axis=0)  # ? vector

    # stop the gradient -- they are meant to be training targets
    sampled_boxes = tf.stop_gradient(ret_boxes, name='sampled_proposal_boxes')
    box_labels = tf.stop_gradient(ret_labels, name='sampled_labels')
    gt_id_for_each_fg = [tf.stop_gradient(fg_inds_wrt_gt) for fg_inds_wrt_gt in all_fg_inds_wrt_gt]

    return sampled_boxes, box_labels, gt_id_for_each_fg


@under_name_scope(name_scope="proposal_metrics")
def proposal_metrics_batch(per_image_ious):
    """
    Add summaries for RPN proposals.
    Args:
         per_image_ious: pairwise intersection-over-union between rpn boxes and gt boxes
                        list of len BS [Num_rpn_boxes x Num_gt_boxes]
    """
    prefix="proposal_metrics_batch"

    # find best roi for each gt, for summary only
    thresholds = [0.3, 0.5]
    best_ious = []
    mean_best_ious = []
    recalls = {}
    for th in thresholds:
        recalls[th] = []

    for batch_index, iou in enumerate(per_image_ious):
        best_iou = tf.reduce_max(iou, axis=0)
        best_ious.append(best_iou)

        mean_best_ious.append(tf.reduce_mean(best_iou))

        # summaries = [mean_best_iou]
        with tf.device('/cpu:0'):
            for th in thresholds:
                recall = tf.truediv(
                    tf.count_nonzero(best_iou >= th),
                    tf.size(best_iou, out_type=tf.int64))
                recalls[th].append(recall)

    all_mean_best_ious = tf.stack(mean_best_ious)
    mean_of_mean_best_iou = tf.reduce_mean(all_mean_best_ious, name='best_iou_per_gt')
    summaries = [mean_of_mean_best_iou]
    for th in thresholds:
        recall = tf.reduce_mean(tf.stack(recalls[th]), name='recall_iou{}'.format(th))
        summaries.append(recall)

    add_moving_summary(*summaries)
