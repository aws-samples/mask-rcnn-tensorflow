# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
# File: data.py

from tensorpack.dataflow import RNGDataFlow
import copy
import numpy as np
import cv2
from tabulate import tabulate
from termcolor import colored

from tensorpack.dataflow import (
    DataFromList, MapDataComponent, MultiProcessMapDataZMQ, MultiThreadMapData, TestDataSpeed, imgaug, MapData)
from tensorpack.utils import logger
from tensorpack.utils.argtools import log_once, memoized

from common import (
    CustomResize, DataFromListOfDict, box_to_point8,
    filter_boxes_inside_shape, point8_to_box, segmentation_to_mask, np_iou)
from config import config as cfg
from dataset import DetectionDataset
from utils.generate_anchors import generate_anchors
from utils.np_box_ops import area as np_area, ioa as np_ioa

import math
# import tensorpack.utils.viz as tpviz

"""
Predefined padding:
Pad all images into one of the 4 predefined padding shapes in cfg.PREPROC.PADDING_SHAPES. Try to batch the images
with same padding shape into the same batch. This can accelerate the data pipeline process.

Functions:
get_padding_shape & _get_padding_shape: Find the closest padding shape to current image according to its aspect ratio
get_next_roidb: Try to get the next image that has the same shape
"""
def _get_padding_shape(aspect_ratio):
    for shape in cfg.PREPROC.PADDING_SHAPES:
        if aspect_ratio >= float(shape[0])/float(shape[1]):
            return shape

    return cfg.PREPROC.PADDING_SHAPES[-1]

def get_padding_shape(h, w):
    aspect_ratio = float(h)/float(w)

    if aspect_ratio > 1.0:
        inv = 1./aspect_ratio
    else:
        inv = aspect_ratio

    shp = _get_padding_shape(inv)
    if aspect_ratio > 1.0:
        return (shp[1], shp[0])
    return shp


def get_next_roidb(roidbs, i, shp, taken):

    if i == len(roidbs) - 1:
        return None

    for k in range(i+1, len(roidbs)):
        if get_padding_shape(roidbs[k]['height'], roidbs[k]['width']) == shp and not taken[k]:
            return k
        if k - i > 40:    # don't try too hard
            break

    # at least try to get one dimension the same
    for k in range(i, len(roidbs)):
        padding_shape = get_padding_shape(roidbs[k]['height'], roidbs[k]['width'])
        if (padding_shape[0] == shp[0] or padding_shape[1] == shp[1]) and not taken[k]:
            return k
        if k - i > 40:    # don't try too hard
            break

    for k in range(i, len(roidbs)):
        if not taken[k]:
            return k

    return None

class DataFromListOfDictBatched(RNGDataFlow):
    def __init__(self, lst, keys, batchsize, shuffle=False):
        self._lst = lst
        self._keys = keys
        self._shuffle = shuffle
        self._size = len(lst)
        self._bs = batchsize

    def __len__(self):
        return int(math.ceil(len(self._lst)/self._bs)) #self._size

    def __iter__(self):
        if self._shuffle:
            self.rng.shuffle(self._lst)
        num_batches = int(math.ceil(len(self._lst)/self._bs))
        for batch in range(num_batches):
            # print(batch)
            last = min(len(self._lst), self._bs*(batch+1))
            dp = [[dic[k] for k in self._keys] for dic in self._lst[batch*self._bs:last]]
            yield dp



class MalformedData(BaseException):
    pass


def print_class_histogram(roidbs):
    """
    Args:
        roidbs (list[dict]): the same format as the output of `load_training_roidbs`.
    """
    dataset = DetectionDataset()
    hist_bins = np.arange(dataset.num_classes + 1)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((dataset.num_classes,), dtype=np.int)
    for entry in roidbs:
        # filter crowd?
        gt_inds = np.where(
            (entry['class'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_classes = entry['class'][gt_inds]
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    data = [[dataset.class_names[i], v] for i, v in enumerate(gt_hist)]
    data.append(['total', sum([x[1] for x in data])])
    table = tabulate(data, headers=['class', '#box'], tablefmt='pipe')
    logger.info("Ground-Truth Boxes:\n" + colored(table, 'cyan'))


@memoized
def get_all_anchors(stride=None, sizes=None, tile=True):
    """
    Get all anchors in the largest possible image, shifted, floatbox
    Args:
        stride (int): the stride of anchors.
        sizes (tuple[int]): the sizes (sqrt area) of anchors

    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.

    """
    if stride is None:
        stride = cfg.RPN.ANCHOR_STRIDE
    if sizes is None:
        sizes = cfg.RPN.ANCHOR_SIZES
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on stride / 2, have (approximate) sqrt areas of the specified
    # sizes, and aspect ratios as given.
    if not cfg.RPN.UNQUANTIZED_ANCHOR:
        cell_anchors = generate_anchors(
            stride,
            scales=np.array(sizes, dtype=np.float) / stride,
            ratios=np.array(cfg.RPN.ANCHOR_RATIOS, dtype=np.float))
    else:
        anchors = []
        ratios=np.array(cfg.RPN.ANCHOR_RATIOS, dtype=np.float)
        for sz in sizes:
            for ratio in ratios:
                w = np.sqrt(sz * sz / ratio)
                h = ratio * w
                anchors.append([-w, -h, w, h])
        cell_anchors = np.asarray(anchors) * 0.5
    # anchors are intbox here.
    # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

    if tile:
        max_size = cfg.PREPROC.MAX_SIZE
        field_size = int(np.ceil(max_size / stride))
        if not cfg.RPN.UNQUANTIZED_ANCHOR: shifts = np.arange(0, field_size) * stride
        else: shifts = (np.arange(0, field_size) * stride).astype("float32")
        shift_x, shift_y = np.meshgrid(shifts, shifts)
        shift_x = shift_x.flatten()
        shift_y = shift_y.flatten()
        shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
        # Kx4, K = field_size * field_size
        K = shifts.shape[0]

        A = cell_anchors.shape[0]
        if not cfg.RPN.UNQUANTIZED_ANCHOR:
            field_of_anchors = (
                cell_anchors.reshape((1, A, 4)) +
                shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        else: field_of_anchors = cell_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
        # FSxFSxAx4
        # Many rounding happens inside the anchor code anyway
        # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
        field_of_anchors = field_of_anchors.astype('float32')
        if not cfg.RPN.UNQUANTIZED_ANCHOR: field_of_anchors[:, :, :, [2, 3]] += 1
        return field_of_anchors
    else:
        cell_anchors = cell_anchors.astype('float32')
        cell_anchors[:, [2, 3]] += 1
        return cell_anchors



@memoized
def get_all_anchors_fpn(strides=None, sizes=None, tile=True):
    """
    Returns:
        foas: anchors for FPN layers p2-p6, each with size S x S x NUM_ANCHOR_RATIOS x 4
        where S == ceil(MAX_SIZE/STRIDE)
    """
    if strides is None:
        strides = cfg.FPN.ANCHOR_STRIDES
    if sizes is None:
        sizes = cfg.RPN.ANCHOR_SIZES
    assert len(strides) == len(sizes)
    foas = []
    for stride, size in zip(strides, sizes):
        foa = get_all_anchors(stride=stride, sizes=(size,), tile=tile)
        foas.append(foa)
    return foas


def get_anchor_labels(anchors, gt_boxes, crowd_boxes):
    """
    Label each anchor as fg/bg/ignore.
    Args:
        anchors: A x 4 float
        gt_boxes: B x 4 float, non-crowd
        crowd_boxes: C x 4 float

    Returns:
        anchor_labels: (A,) int. Each element is {-1, 0, 1}
        anchor_boxes: A x 4. Contains the target gt_box for each anchor when the anchor is fg.
    """
    # This function will modify labels and return the filtered inds
    def filter_box_label(labels, value, max_num):
        curr_inds = np.where(labels == value)[0]
        if len(curr_inds) > max_num:
            np.random.seed(cfg.TRAIN.SEED)
            disable_inds = np.random.choice(
                curr_inds, size=(len(curr_inds) - max_num),
                replace=False)
            labels[disable_inds] = -1    # ignore them
            curr_inds = np.where(labels == value)[0]
        return curr_inds

    NA, NB = len(anchors), len(gt_boxes)
    assert NB > 0  # empty images should have been filtered already
    box_ious = np_iou(anchors, gt_boxes)  # NA x NB
    ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA,
    ious_max_per_anchor = box_ious.max(axis=1)
    ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB
    # for each gt, find all those anchors (including ties) that has the max ious with it
    anchors_with_max_iou_per_gt = np.where(box_ious == ious_max_per_gt)[0]

    # Setting NA labels: 1--fg 0--bg -1--ignore
    anchor_labels = -np.ones((NA,), dtype='int32')   # NA,

    # the order of setting neg/pos labels matter
    anchor_labels[anchors_with_max_iou_per_gt] = 1
    anchor_labels[ious_max_per_anchor >= cfg.RPN.POSITIVE_ANCHOR_THRESH] = 1
    anchor_labels[ious_max_per_anchor < cfg.RPN.NEGATIVE_ANCHOR_THRESH] = 0

    # label all non-ignore candidate boxes which overlap crowd as ignore
    if crowd_boxes.size > 0:
        cand_inds = np.where(anchor_labels >= 0)[0]
        cand_anchors = anchors[cand_inds]
        ioas = np_ioa(crowd_boxes, cand_anchors)
        overlap_with_crowd = cand_inds[ioas.max(axis=0) > cfg.RPN.CROWD_OVERLAP_THRESH]
        anchor_labels[overlap_with_crowd] = -1

    # Subsample fg labels: ignore some fg if fg is too many
    target_num_fg = int(cfg.RPN.BATCH_PER_IM * cfg.RPN.FG_RATIO)
    fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)
    # Keep an image even if there is no foreground anchors
    # if len(fg_inds) == 0:
    #     raise MalformedData("No valid foreground for RPN!")

    # Subsample bg labels. num_bg is not allowed to be too many
    old_num_bg = np.sum(anchor_labels == 0)
    if old_num_bg == 0:
        # No valid bg in this image, skip.
        raise MalformedData("No valid background for RPN!")
    target_num_bg = cfg.RPN.BATCH_PER_IM - len(fg_inds)
    filter_box_label(anchor_labels, 0, target_num_bg)   # ignore return values

    # Set anchor boxes: the best gt_box for each fg anchor
    anchor_boxes = np.zeros((NA, 4), dtype='float32')
    fg_boxes = gt_boxes[ious_argmax_per_anchor[fg_inds], :]
    anchor_boxes[fg_inds, :] = fg_boxes
    # assert len(fg_inds) + np.sum(anchor_labels == 0) == cfg.RPN.BATCH_PER_IM
    return anchor_labels, anchor_boxes


def get_rpn_anchor_input(im, boxes, is_crowd):
    """
    Args:
        im: an image
        boxes: nx4, floatbox, gt. shoudn't be changed
        is_crowd: n,

    Returns:
        The anchor labels and target boxes for each pixel in the featuremap.
        fm_labels: fHxfWxNA
        fm_boxes: fHxfWxNAx4
        NA will be NUM_ANCHOR_SIZES x NUM_ANCHOR_RATIOS
    """
    boxes = boxes.copy()
    all_anchors = np.copy(get_all_anchors())
    # fHxfWxAx4 -> (-1, 4)
    featuremap_anchors_flatten = all_anchors.reshape((-1, 4))

    # only use anchors inside the image
    inside_ind, inside_anchors = filter_boxes_inside_shape(featuremap_anchors_flatten, im.shape[:2])
    # obtain anchor labels and their corresponding gt boxes
    anchor_labels, anchor_gt_boxes = get_anchor_labels(inside_anchors, boxes[is_crowd == 0], boxes[is_crowd == 1])

    # Fill them back to original size: fHxfWx1, fHxfWx4
    anchorH, anchorW = all_anchors.shape[:2]
    featuremap_labels = -np.ones((anchorH * anchorW * cfg.RPN.NUM_ANCHOR, ), dtype='int32')
    featuremap_labels[inside_ind] = anchor_labels
    featuremap_labels = featuremap_labels.reshape((anchorH, anchorW, cfg.RPN.NUM_ANCHOR))
    featuremap_boxes = np.zeros((anchorH * anchorW * cfg.RPN.NUM_ANCHOR, 4), dtype='float32')
    featuremap_boxes[inside_ind, :] = anchor_gt_boxes
    featuremap_boxes = featuremap_boxes.reshape((anchorH, anchorW, cfg.RPN.NUM_ANCHOR, 4))
    return featuremap_labels, featuremap_boxes


def get_multilevel_rpn_anchor_input(im, boxes, is_crowd):
    """
    Args:
        im: a single image, H_image x W_image x NumChannel
        boxes: n x 4, floatbox, gt. shoudn't be changed
        is_crowd: (n,), for each box, is it crowd

    Returns:
        [(fm_labels, fm_boxes)]: Returns a tuple for each FPN level.
        Each tuple contains the anchor labels and target boxes for each pixel in the featuremap.

        fm_labels: H_feature x W_feature x NUM_ANCHOR_RATIOS
        fm_boxes: H_feature x W_feature x NUM_ANCHOR_RATIOS x4
    """
    boxes = boxes.copy()
    anchors_per_level = get_all_anchors_fpn()
    flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
    all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)

    inside_ind, inside_anchors = filter_boxes_inside_shape(all_anchors_flatten, im.shape[:2])
    anchor_labels, anchor_gt_boxes = get_anchor_labels(inside_anchors, boxes[is_crowd == 0], boxes[is_crowd == 1])

    # map back to all_anchors, then split to each level
    num_all_anchors = all_anchors_flatten.shape[0]
    all_labels = -np.ones((num_all_anchors, ), dtype='int32')
    all_labels[inside_ind] = anchor_labels
    all_boxes = np.zeros((num_all_anchors, 4), dtype='float32')
    all_boxes[inside_ind] = anchor_gt_boxes

    start = 0
    multilevel_inputs = []
    for level_anchor in anchors_per_level:
        assert level_anchor.shape[2] == len(cfg.RPN.ANCHOR_RATIOS)
        anchor_shape = level_anchor.shape[:3]   # fHxfWxNUM_ANCHOR_RATIOS
        num_anchor_this_level = np.prod(anchor_shape)
        end = start + num_anchor_this_level
        multilevel_inputs.append(
            (all_labels[start: end].reshape(anchor_shape),
             all_boxes[start: end, :].reshape(anchor_shape + (4,))
             ))
        start = end
    assert end == num_all_anchors, "{} != {}".format(end, num_all_anchors)
    return multilevel_inputs


def get_train_dataflow():
    """
    Return a training dataflow. Each datapoint consists of the following:

    An image: (h, w, 3),

    1 or more pairs of (anchor_labels, anchor_boxes):
    anchor_labels: (h', w', NA)
    anchor_boxes: (h', w', NA, 4)

    gt_boxes: (N, 4)
    gt_labels: (N,)

    If MODE_MASK, gt_masks: (N, h, w)
    """

    roidbs = DetectionDataset().load_training_roidbs(cfg.DATA.TRAIN)
    print_class_histogram(roidbs)

    # Valid training images should have at least one fg box.
    # But this filter shall not be applied for testing.
    num = len(roidbs)
    roidbs = list(filter(lambda img: len(img['boxes'][img['is_crowd'] == 0]) > 0, roidbs))
    logger.info("Filtered {} images which contain no non-crowd groudtruth boxes. Total #images for training: {}".format(
        num - len(roidbs), len(roidbs)))

    ds = DataFromList(roidbs, shuffle=True)

    aug = imgaug.AugmentorList(
        [CustomResize(cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE),
         imgaug.Flip(horiz=True)])

    def preprocess(roidb):
        fname, boxes, klass, is_crowd = roidb['file_name'], roidb['boxes'], roidb['class'], roidb['is_crowd']
        boxes = np.copy(boxes)
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        im = im.astype('float32')
        # assume floatbox as input
        assert boxes.dtype == np.float32, "Loader has to return floating point boxes!"

        # augmentation:
        im, params = aug.augment_return_params(im)
        points = box_to_point8(boxes)
        points = aug.augment_coords(points, params)
        boxes = point8_to_box(points)
        assert np.min(np_area(boxes)) > 0, "Some boxes have zero area!"

        ret = {'images': im}
        # rpn anchor:
        try:
            if cfg.MODE_FPN:
                multilevel_anchor_inputs = get_multilevel_rpn_anchor_input(im, boxes, is_crowd)
                for i, (anchor_labels, anchor_boxes) in enumerate(multilevel_anchor_inputs):
                    ret['anchor_labels_lvl{}'.format(i + 2)] = anchor_labels
                    ret['anchor_boxes_lvl{}'.format(i + 2)] = anchor_boxes
            else:
                # anchor_labels, anchor_boxes
                ret['anchor_labels'], ret['anchor_boxes'] = get_rpn_anchor_input(im, boxes, is_crowd)

            boxes = boxes[is_crowd == 0]    # skip crowd boxes in training target
            klass = klass[is_crowd == 0]
            ret['gt_boxes'] = boxes
            ret['gt_labels'] = klass
            if not len(boxes):
                raise MalformedData("No valid gt_boxes!")
        except MalformedData as e:
            log_once("Input {} is filtered for training: {}".format(fname, str(e)), 'warn')
            return None

        if cfg.MODE_MASK:
            # augmentation will modify the polys in-place
            segmentation = copy.deepcopy(roidb['segmentation'])
            segmentation = [segmentation[k] for k in range(len(segmentation)) if not is_crowd[k]]
            assert len(segmentation) == len(boxes)

            # Apply augmentation on polygon coordinates.
            # And produce one image-sized binary mask per box.
            masks = []
            for polys in segmentation:
                polys = [aug.augment_coords(p, params) for p in polys]
                masks.append(segmentation_to_mask(polys, im.shape[0], im.shape[1]))
            masks = np.asarray(masks, dtype='uint8')    # values in {0, 1}
            ret['gt_masks'] = masks

            # from viz import draw_annotation, draw_mask
            # viz = draw_annotation(im, boxes, klass)
            # for mask in masks:
            #     viz = draw_mask(viz, mask)
            # tpviz.interactive_imshow(viz)

        # for k, v in ret.items():
        #    print("key", k)
        #    if type(v) == np.ndarray:
        #        print("val", v.shape)
        #    else:
        #        print("val", v)

        return ret

    if cfg.TRAINER == 'horovod':
        #ds = MapData(ds, preprocess)
        ds = MultiThreadMapData(ds, 5, preprocess)
        # MPI does not like fork()
    else:
        ds = MultiProcessMapDataZMQ(ds, 10, preprocess)
    return ds


def get_batch_train_dataflow(batch_size):
    """
    Return a training dataflow. Each datapoint consists of the following:

    A batch of images: (BS, h, w, 3),

    For each image

    1 or more pairs of (anchor_labels, anchor_boxes) :
    anchor_labels: (BS, h', w', maxNumAnchors)
    anchor_boxes: (BS, h', w', maxNumAnchors, 4)

    gt_boxes: (BS, maxNumAnchors, 4)
    gt_labels: (BS, maxNumAnchors)

    If MODE_MASK, gt_masks: (BS, maxNumAnchors, h, w)
    """
    print("In train dataflow")
    roidbs = DetectionDataset().load_training_roidbs(cfg.DATA.TRAIN)
    print("Done loading roidbs")

    # print_class_histogram(roidbs)

    # Valid training images should have at least one fg box.
    # But this filter shall not be applied for testing.
    num = len(roidbs)
    roidbs = list(filter(lambda img: len(img['boxes'][img['is_crowd'] == 0]) > 0, roidbs))
    logger.info("Filtered {} images which contain no non-crowd groudtruth boxes. Total #images for training: {}".format(
        num - len(roidbs), len(roidbs)))

    roidbs = sorted(roidbs, key=lambda x: float(x['width']) / float(x['height']), reverse=True)     # will shuffle it later at every rank

    print("Batching roidbs")
    batched_roidbs = []

    if cfg.PREPROC.PREDEFINED_PADDING:
        taken = [False for _ in roidbs]
        done = False

        for i, d in enumerate(roidbs):
            batch = []
            if not taken[i]:
                batch.append(d)
                padding_shape = get_padding_shape(d['height'], d['width'])
                while len(batch) < batch_size:
                    k = get_next_roidb(roidbs, i, padding_shape, taken)
                    if k == None:
                        done = True
                        break
                    batch.append(roidbs[k])
                    taken[i], taken[k] = True, True
                if not done:
                    batched_roidbs.append(batch)
    else:
        batch = []
        for i, d in enumerate(roidbs):
            if i % batch_size == 0:
                if len(batch) == batch_size:
                    batched_roidbs.append(batch)
                batch = []
            batch.append(d)

    #batched_roidbs = sort_by_aspect_ratio(roidbs, batch_size)
    #batched_roidbs = group_by_aspect_ratio(roidbs, batch_size)
    print("Done batching roidbs")


    # Notes:
    #   - discard any leftover images
    #   - The batches will be shuffled, but the contents of each batch will always be the same
    #   - TODO: Fix lack of batch contents shuffling


    aug = imgaug.AugmentorList(
         [CustomResize(cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE),
          imgaug.Flip(horiz=True)])

    # aug = imgaug.AugmentorList([CustomResize(cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)])


    def preprocess(roidb_batch):
        datapoint_list = []
        for roidb in roidb_batch:
            fname, boxes, klass, is_crowd = roidb['file_name'], roidb['boxes'], roidb['class'], roidb['is_crowd']
            boxes = np.copy(boxes)
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert im is not None, fname
            im = im.astype('float32')
            # assume floatbox as input
            assert boxes.dtype == np.float32, "Loader has to return floating point boxes!"

            # augmentation:
            im, params = aug.augment_return_params(im)
            points = box_to_point8(boxes)
            points = aug.augment_coords(points, params)
            boxes = point8_to_box(points)
            assert np.min(np_area(boxes)) > 0, "Some boxes have zero area!"

            ret = {'images': im}
            # rpn anchor:
            try:
                if cfg.MODE_FPN:
                    multilevel_anchor_inputs = get_multilevel_rpn_anchor_input(im, boxes, is_crowd)
                    for i, (anchor_labels, anchor_boxes) in enumerate(multilevel_anchor_inputs):
                        ret['anchor_labels_lvl{}'.format(i + 2)] = anchor_labels
                        ret['anchor_boxes_lvl{}'.format(i + 2)] = anchor_boxes
                else:
                    raise NotImplementedError("Batch mode only available for FPN")

                boxes = boxes[is_crowd == 0]    # skip crowd boxes in training target
                klass = klass[is_crowd == 0]
                ret['gt_boxes'] = boxes
                ret['gt_labels'] = klass
                ret['filename'] = fname
                if not len(boxes):
                    raise MalformedData("No valid gt_boxes!")
            except MalformedData as e:
                log_once("Input {} is filtered for training: {}".format(fname, str(e)), 'warn')
                return None

            if cfg.MODE_MASK:
                # augmentation will modify the polys in-place
                segmentation = copy.deepcopy(roidb['segmentation'])
                segmentation = [segmentation[k] for k in range(len(segmentation)) if not is_crowd[k]]
                assert len(segmentation) == len(boxes)

                # Apply augmentation on polygon coordinates.
                # And produce one image-sized binary mask per box.
                masks = []
                for polys in segmentation:
                    polys = [aug.augment_coords(p, params) for p in polys]
                    masks.append(segmentation_to_mask(polys, im.shape[0], im.shape[1]))
                masks = np.asarray(masks, dtype='uint8')    # values in {0, 1}
                ret['gt_masks'] = masks

            datapoint_list.append(ret)

        #################################################################################################################
        # Batchify the output
        #################################################################################################################

        # Now we need to batch the various fields

        # Easily stackable:
        # - anchor_labels_lvl2
        # - anchor_boxes_lvl2
        # - anchor_labels_lvl3
        # - anchor_boxes_lvl3
        # - anchor_labels_lvl4
        # - anchor_boxes_lvl4
        # - anchor_labels_lvl5
        # - anchor_boxes_lvl5
        # - anchor_labels_lvl6
        # - anchor_boxes_lvl6

        batched_datapoint = {}
        for stackable_field in ["anchor_labels_lvl2",
                                "anchor_boxes_lvl2",
                                "anchor_labels_lvl3",
                                "anchor_boxes_lvl3",
                                "anchor_labels_lvl4",
                                "anchor_boxes_lvl4",
                                "anchor_labels_lvl5",
                                "anchor_boxes_lvl5",
                                "anchor_labels_lvl6",
                                "anchor_boxes_lvl6"]:
            batched_datapoint[stackable_field] = np.stack([d[stackable_field] for d in datapoint_list])



        # Require padding and original dimension storage
        # - image (HxWx3)
        # - gt_boxes (?x4)
        # - gt_labels (?)
        # - gt_masks (?xHxW)

        """
        Find the minimum container size for images (maxW x maxH)
        Find the maximum number of ground truth boxes
        For each image, save original dimension and pad
        """

        if cfg.PREPROC.PREDEFINED_PADDING:
            padding_shapes = [get_padding_shape(*(d["images"].shape[:2])) for d in datapoint_list]
            max_height = max([shp[0] for shp in padding_shapes])
            max_width = max([shp[1] for shp in padding_shapes])
        else:
            image_dims = [d["images"].shape for d in datapoint_list]
            heights = [dim[0] for dim in image_dims]
            widths = [dim[1] for dim in image_dims]

            max_height = max(heights)
            max_width = max(widths)


        # image
        padded_images = []
        original_image_dims = []
        for datapoint in datapoint_list:
            image = datapoint["images"]
            original_image_dims.append(image.shape)

            h_padding = max_height - image.shape[0]
            w_padding = max_width - image.shape[1]

            padded_image = np.pad(image,
                                  [[0, h_padding],
                                   [0, w_padding],
                                   [0, 0]],
                                  'constant')

            padded_images.append(padded_image)

        batched_datapoint["images"] = np.stack(padded_images)
        #print(batched_datapoint["images"].shape)
        batched_datapoint["orig_image_dims"] = np.stack(original_image_dims)


        # gt_boxes and gt_labels
        max_num_gts = max([d["gt_labels"].size for d in datapoint_list])

        gt_counts = []
        padded_gt_labels = []
        padded_gt_boxes = []
        padded_gt_masks = []
        for datapoint in datapoint_list:
            gt_count_for_image = datapoint["gt_labels"].size
            gt_counts.append(gt_count_for_image)

            gt_padding = max_num_gts - gt_count_for_image

            padded_gt_labels_for_img = np.pad(datapoint["gt_labels"], [0, gt_padding], 'constant', constant_values=-1)
            padded_gt_labels.append(padded_gt_labels_for_img)

            padded_gt_boxes_for_img = np.pad(datapoint["gt_boxes"],
                                             [[0, gt_padding],
                                              [0,0]],
                                             'constant')
            padded_gt_boxes.append(padded_gt_boxes_for_img)




            h_padding = max_height - datapoint["images"].shape[0]
            w_padding = max_width - datapoint["images"].shape[1]



            if cfg.MODE_MASK:
                padded_gt_masks_for_img = np.pad(datapoint["gt_masks"],
                                         [[0, gt_padding],
                                          [0, h_padding],
                                          [0, w_padding]],
                                         'constant')
                padded_gt_masks.append(padded_gt_masks_for_img)


        batched_datapoint["orig_gt_counts"] = np.stack(gt_counts)
        batched_datapoint["gt_labels"] = np.stack(padded_gt_labels)
        batched_datapoint["gt_boxes"] = np.stack(padded_gt_boxes)
        batched_datapoint["filenames"] = [d["filename"] for d in datapoint_list]

        if cfg.MODE_MASK:
            batched_datapoint["gt_masks"] = np.stack(padded_gt_masks)



        return batched_datapoint

    ds = DataFromList(batched_roidbs, shuffle=True)



    if cfg.TRAINER == 'horovod':
        # ds = MapData(ds, preprocess)
        ds = MultiThreadMapData(ds, 5, preprocess)
        # MPI does not like fork()
    else:
        ds = MultiProcessMapDataZMQ(ds, 10, preprocess)
    return ds


def get_eval_dataflow(name, shard=0, num_shards=1):
    """
    Args:
        name (str): name of the dataset to evaluate
        shard, num_shards: to get subset of evaluation data
    """
    roidbs = DetectionDataset().load_inference_roidbs(name)

    num_imgs = len(roidbs)
    img_per_shard = num_imgs // num_shards
    img_range = (shard * img_per_shard, (shard + 1) * img_per_shard if shard + 1 < num_shards else num_imgs)

    # no filter for training
    ds = DataFromListOfDict(roidbs[img_range[0]: img_range[1]], ['file_name', 'id'])

    def f(fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        return im
    ds = MapDataComponent(ds, f, 0)
    # Evaluation itself may be multi-threaded, therefore don't add prefetch here.
    return ds


def get_batched_eval_dataflow(name, shard=0, num_shards=1, batch_size=1):
    """
    Args:
        name (str): name of the dataset to evaluate
        shard, num_shards: to get subset of evaluation data
    """
    roidbs = DetectionDataset().load_inference_roidbs(name)

    num_imgs = len(roidbs)
    img_per_shard = num_imgs // num_shards
    img_range = (shard * img_per_shard, (shard + 1) * img_per_shard if shard + 1 < num_shards else num_imgs)

    # no filter for training
    ds = DataFromListOfDictBatched(roidbs[img_range[0]: img_range[1]], ['file_name', 'image_id'], batch_size)

    def decode_images(inputs):
        return [[cv2.imread(inp[0], cv2.IMREAD_COLOR), inp[1]] for inp in inputs]

    def resize_images(inputs):
        resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
        resized_imgs = [resizer.augment(inp[0]) for inp in inputs]
        org_shapes = [inp[0].shape for inp in inputs]
        scales = [np.sqrt(rimg.shape[0] * 1.0 / org_shape[0] * rimg.shape[1] / org_shape[1]) for rimg, org_shape in zip(resized_imgs, org_shapes)]

        return [[resized_imgs[i], inp[1], scales[i], org_shapes[i][:2]] for i, inp in enumerate(inputs)]

    def pad_and_batch(inputs):
        heights, widths, _ = zip(*[inp[0].shape for inp in inputs])
        max_h, max_w = max(heights), max(widths)
        padded_images = np.stack([np.pad(inp[0], [[0, max_h-inp[0].shape[0]], [0, max_w-inp[0].shape[1]], [0,0]], 'constant') for inp in inputs])
        return [padded_images, [inp[1] for inp in inputs], list(zip(heights, widths)), [inp[2] for inp in inputs], [inp[3] for inp in inputs]]

    ds = MapData(ds, decode_images)
    ds = MapData(ds, resize_images)
    ds = MapData(ds, pad_and_batch)
    return ds


if __name__ == '__main__':
    import os
    from tensorpack.dataflow import PrintData
    cfg.DATA.BASEDIR = os.path.expanduser('~/data')
    ds = get_batched_eval_dataflow('train_reduced')
    ds.reset_state()
    cnt = 0
    for k in ds:
        print(k)
        cnt += 1
