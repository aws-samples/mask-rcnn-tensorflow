# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
# File: viz.py

import numpy as np
from six.moves import zip

from tensorpack.utils import viz
from tensorpack.utils.palette import PALETTE_RGB

from config import config as cfg
from utils.np_box_ops import iou as np_iou
from eval import DetectionResult
import cv2


def draw_annotation(img, boxes, klass, is_crowd=None):
    """Will not modify img"""
    labels = []
    assert len(boxes) == len(klass)
    if is_crowd is not None:
        assert len(boxes) == len(is_crowd)
        for cls, crd in zip(klass, is_crowd):
            clsname = cfg.DATA.CLASS_NAMES[cls]
            if crd == 1:
                clsname += ';Crowd'
            labels.append(clsname)
    else:
        for cls in klass:
            labels.append(cfg.DATA.CLASS_NAMES[cls])
    img = viz.draw_boxes(img, boxes, labels)
    return img


def draw_proposal_recall(img, proposals, gt_boxes, top=3):
    """
    Draw top3 proposals for each gt.
    Args:
        proposals: NPx4
        proposal_scores: NP
        gt_boxes: NG
    """
    box_ious = np_iou(gt_boxes, proposals)    # ng x np
    box_ious_argsort = np.argsort(-box_ious, axis=1)
    good_proposals_ind = box_ious_argsort[:, :top]   # for each gt, find 3 best proposals
    good_proposals_ind = np.unique(good_proposals_ind.ravel())

    proposals = proposals[good_proposals_ind, :]
    img = viz.draw_boxes(img, proposals)
    return img, good_proposals_ind


def draw_predictions(img, boxes, scores):
    """
    Args:
        boxes: kx4
        scores: kxC
    """
    if len(boxes) == 0:
        return img
    labels = scores.argmax(axis=1)
    scores = scores.max(axis=1)
    tags = ["{},{:.2f}".format(cfg.DATA.CLASS_NAMES[lb], score) for lb, score in zip(labels, scores)]
    return viz.draw_boxes(img, boxes, tags)


def draw_final_outputs(img, results):
    """
    Args:
        results: [DetectionResult]
    """
    if len(results) == 0:
        return img

    tags = []
    for r in results:
        tags.append(
            "{},{:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score))
    boxes = np.asarray([r.box for r in results])
    ret = viz.draw_boxes(img, boxes, tags)

    for r in results:
        if r.mask is not None:
            ret = draw_mask(ret, r.mask)
    return ret


def draw_outputs(img, final_boxes, final_scores, final_labels, threshold=0.8):
    """
    Args:
        results: [DetectionResult]
    """
    results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels)) if args[1]>threshold]
    if len(results) == 0:
        return img

    tags = []
    for r in results:
        tags.append(
            "{},{:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score))
    boxes = np.asarray([r.box for r in results])
    ret = viz.draw_boxes(img, boxes, tags)

    for r in results:
        if r.mask is not None:
            ret = draw_mask(ret, r.mask)
    return ret


def draw_mask(im, mask, alpha=0.5, color=None):
    """
    Overlay a mask on top of the image.

    Args:
        im: a 3-channel uint8 image in BGR
        mask: a binary 1-channel image of the same size
        color: if None, will choose automatically
    """
    if color is None:
        color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]
    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  im * (1 - alpha) + color * alpha, im)
    im = im.astype('uint8')
    return im


def get_mask(img, box, mask, threshold=.5):
    box = box.astype(int)
    color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]
    a_mask = np.stack([(cv2.resize(mask, (box[2]-box[0], box[3]-box[1])) > threshold).astype(np.int8)]*3, axis=2)
    sub_image = img[box[1]:box[3],box[0]:box[2],:].astype(np.uint8)
    sub_image = np.where(a_mask==1, sub_image * (1 - 0.5) + color * 0.5, sub_image)
    new_image = img.copy()
    new_image[box[1]:box[3],box[0]:box[2],:] = sub_image
    return new_image


def apply_masks(img, boxes, masks, scores, score_threshold=.7, mask_threshold=0.5):
    image = img.copy()
    for i,j,k in zip(boxes, masks, scores):
        if k>= score_threshold:
            image = get_mask(image, i, j, mask_threshold)
    return image


def gt_mask(img, masks):
    new_image = img.copy()
    for mask in masks:
        color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]
        a_mask = np.stack([mask.astype(np.int8)]*3, axis=2)
        new_image = np.where(a_mask==1, new_image * (1 - 0.5) + color * 0.5, new_image)
    return new_image
