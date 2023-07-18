# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import tensorflow as tf
from MaskRCNN.performance import print_runtime_shape, print_runtime_tensor

from tensorpack.tfutils.scope_utils import under_name_scope

from config import config

@under_name_scope()
def clip_boxes_batch(boxes, image_hw, name=None):
    """
    Args:
        boxes: BS X N X (#class) X 4 (x1y1x2y2)
        image_hw: BSx2 
    Returns:
        BS X N X (#class) X 4 (x1y1x2y2)
    """
    boxes_shape = tf.shape(boxes)
    boxes = tf.maximum(boxes, 0.0)

    image_wh = tf.reverse(image_hw, axis=[-1])
    whwh = tf.concat((image_wh, image_wh), axis=1)
    whwh = tf.expand_dims(whwh, 1)
    whwh = tf.expand_dims(whwh, 1)

    whwh_tiled = tf.tile(whwh, [1, boxes_shape[1], boxes_shape[2], 1])    # BS X N X (#class) X 4
    boxes = tf.minimum(boxes, tf.cast(whwh_tiled, tf.float32), name=name)
    return boxes

@under_name_scope()
def clip_boxes(boxes, window, name=None):
    """
    Args:
        boxes: nx4, xyxy
        window: [h, w]
    """
    window = tf.squeeze(window, axis=0)

    boxes = tf.maximum(boxes, 0.0)
    m = tf.tile(tf.reverse(window, [0]), [2])    # (4,)
    boxes = tf.minimum(boxes, tf.cast(m, tf.float32), name=name)
    return boxes


@under_name_scope()
def decode_bbox_target(box_predictions, anchors):
    """
    Args:
        box_predictions: BS X N X 4(tx, ty, tw, th)
        anchors: BS X N X 4 floatbox. Must have the same shape (x1, y1, x2, y2)

    Returns:
        box_decoded: BS X N X 4 (x1, y1, x2, y2)
    """
    orig_shape = tf.shape(anchors)
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
    # each is (...)x1x2
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    clip = tf.cast(tf.math.log(config.PREPROC.MAX_SIZE / 16.), dtype=tf.float32)
    wbhb = tf.math.exp(tf.minimum(box_pred_twth, clip)) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5    
    out = tf.concat([x1y1, x2y2], axis=-2)
    return tf.reshape(out, orig_shape)


@under_name_scope()
def encode_bbox_target(boxes, anchors):
    """
    Args:
        boxes: (..., 4), float32 (x1, y1, x2, y2)
        anchors: (..., 4), float32 (x1, y1, x2, y2)

    Returns:
        box_encoded: (..., 4), (tx, ty, tw, th)
    """
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)
    anchors_wh = anchors_x2y2 - anchors_x1y1
    anchors_xy = (anchors_x2y2 + anchors_x1y1) * 0.5

    boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
    boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, 2, axis=1)
    boxes_wh = boxes_x2y2 - boxes_x1y1
    boxes_xy = (boxes_x2y2 + boxes_x1y1) * 0.5

    # Note that here not all boxes are valid. Some may be zero
    txty = (boxes_xy - anchors_xy) / anchors_wh
    twth = tf.math.log(boxes_wh / anchors_wh)  # may contain -inf for invalid boxes
    encoded = tf.concat([txty, twth], axis=1)  
    encoded = tf.reshape(encoded, tf.shape(boxes))
    return encoded


@under_name_scope()
def crop_and_resize(image, boxes, box_ind, crop_size, orig_image_dims, pad_border=True):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

    Args:
        image: N x H x W x C
        boxes: nx4, y1x1y2x2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    assert isinstance(crop_size, int), crop_size
    
    #image = print_runtime_shape("image", image)
    #boxes = print_runtime_shape("boxes", boxes)
    #box_ind = print_runtime_shape("box_ind", box_ind)
    
    boxes = tf.stop_gradient(boxes)
    image = image[:, :orig_image_dims[0], :orig_image_dims[1], :]

    if pad_border:
        image = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
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
        y0, x0, y1, x1 = tf.split(boxes, 4, axis=-1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    image_shape = tf.shape(image)[1:3]

    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    ret = tf.image.crop_and_resize(image, boxes, tf.cast(box_ind, tf.int32), 
        crop_size=[crop_size, crop_size])
    return ret

@under_name_scope()
def permute_boxes_coords(boxes):
  """
    Args:
      boxes Tensor where last axis is of dimension 4 
    Returns:
      boxes: Tensor where last axis is of dimension 4 
  """
  boxes = tf.unstack(boxes, axis=-1)
  return tf.stack([ boxes[1], boxes[0], boxes[3], boxes[2] ], axis=-1)

class RPNGroundTruth(namedtuple('_RPNGroundTruth', ['boxes', 'gt_labels', 'gt_boxes'])):
    """
    boxes (FS x FS x NA x 4): The anchor boxes.
    gt_labels (BS X FS x FS x NA):
    gt_boxes (BS X FS x FS x NA x 4): Groundtruth boxes corresponding to each anchor.
    """
    def encoded_gt_boxes(self):
        gt_boxes = tf.unstack(self.gt_boxes, num=config.TRAIN.BATCH_SIZE_PER_GPU)
        return tf.stack([encode_bbox_target(image_gt_boxes, self.boxes) for  image_gt_boxes in gt_boxes])

    def decode_logits(self, logits):
        pred_boxes = tf.unstack(logits, num=config.TRAIN.BATCH_SIZE_PER_GPU)
        return tf.stack([decode_bbox_target(image_pred_boxes, self.boxes) for  image_pred_boxes in  pred_boxes])