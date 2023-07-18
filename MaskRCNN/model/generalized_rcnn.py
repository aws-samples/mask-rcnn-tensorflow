# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import six

from tensorpack.utils.argtools import memoized

assert six.PY3, "FasterRCNN requires Python 3!"
import tensorflow as tf

from tensorpack.models.regularize import regularize_cost, l2_regularizer
from tensorpack.tfutils.summary import add_moving_summary, get_current_tower_context
from tensorpack.graph_builder.model_desc import ModelDesc
from tensorpack.tfutils.common import tfv1
from tensorpack.tfutils.summary import add_moving_summary

from model import mask_head, boxclass_head
from model.backbone import image_preprocess, resnet_fpn_backbone
from config import config as cfg
from data import get_all_anchors_fpn
from model_box import RPNGroundTruth, clip_boxes_batch, crop_and_resize, permute_boxes_coords
from model.fpn import fpn_model
from model.boxclass_head import boxclass_predictions, boxclass_outputs, BoxClassHead
from model.biased_sampler import sample_fast_rcnn_targets
from model.mask_head import maskrcnn_loss
from model.rpn import rpn_head, generate_fpn_proposals, batch_rpn_losses
from model.roi_ops import roi_features, roi_level_summary
from utils.randomnness import SeedGenerator

from performance import print_runtime_shape, print_runtime_tensor


class GradientClipOptimizer(tfv1.train.Optimizer):
    def __init__(self, opt, clip_norm):
        self.opt = opt
        self.clip_norm = clip_norm

    def compute_gradients(self, *args, **kwargs):
        return self.opt.compute_gradients(*args, **kwargs)
    """
    def apply_gradients(self, *args, **kwargs):
        return self.opt.apply_gradients(*args, **kwargs)
    """
    def apply_gradients(self, gradvars, global_step=None, name=None):
        old_grads, v = zip(*gradvars)
        all_are_finite = tf.math.reduce_all([tf.math.reduce_all(tf.math.is_finite(g)) for g in old_grads])
        clipped_grads, _ = tf.clip_by_global_norm(old_grads, self.clip_norm,
                                         use_norm=tf.cond(
                                         all_are_finite,
                                         lambda: tf.linalg.global_norm(old_grads),
                                         lambda: tf.constant(self.clip_norm, dtype=tf.float32)), name='clip_by_global_norm')
        gradvars = list(zip(clipped_grads, v))
        return self.opt.apply_gradients(gradvars, global_step, name)

    def get_slot(self, *args, **kwargs):
        return self.opt.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self.opt.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self.opt.variables(*args, **kwargs)

def nchw_to_nhwc_transform(input):
    return tf.transpose(input, [0, 2, 3, 1])

def nhwc_to_nchw_transform(input):
    return tf.transpose(input, [0, 3, 1, 2])

class DetectionModel(ModelDesc):
    def __init__(self):
       pass

    def preprocess(self, image):
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2]) if cfg.TRAIN.BACKBONE_NCHW else image

    @property
    def training(self):
        return get_current_tower_context().is_training

    def optimizer(self):
        lr = tf.compat.v1.get_variable ('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)

        opt = tf.compat.v1.train.MomentumOptimizer(lr, 0.9)
        if cfg.TRAIN.GRADIENT_CLIP != 0:
            opt = GradientClipOptimizer(opt, cfg.TRAIN.GRADIENT_CLIP)
        return opt

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """

        out = ['output/batch_indices', 'output/boxes', 'output/scores', 'output/labels']

        if cfg.MODE_MASK:
            out.append('output/masks')
        return ['images', 'orig_image_dims'], out

    def build_graph(self, *inputs):
        #tf.debugging.enable_check_numerics()

        inputs = dict(zip(self.input_names, inputs))

        images = self.preprocess(inputs['images'])   
        seed_gen = SeedGenerator(cfg.TRAIN.SEED)

        p_features = self.fpn_features(images, seed_gen) # features for levels p2, p3, p4, p5 p6

        anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}
        image_shape2d = tf.shape(images)[2:] if cfg.TRAIN.BACKBONE_NCHW else tf.shape(images)[1:3] 
        proposal_rois, rpn_losses = self.rpn(image_shape2d, p_features, anchor_inputs, inputs['orig_image_dims'], seed_gen)

        targets = [inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]
        head_losses = self.roi_heads(p_features, proposal_rois, targets, inputs, seed_gen)

        if self.training:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            total_cost = tf.add_n(rpn_losses + head_losses + [wd_cost], 'total_cost')
            add_moving_summary(total_cost, wd_cost)
            return total_cost

class ResNetFPNModel(DetectionModel):
    def __init__(self):
        super(ResNetFPNModel, self).__init__()

    def inputs(self):

        ret = [
            tfv1.placeholder(tf.string, (None,), 'filenames'), # N length vector of filenames
            tfv1.placeholder(tf.float32, (None, None, None, 3), 'images'),  # N x H x W x C
            tfv1.placeholder(tf.int32, (None, 3), 'orig_image_dims')  # N x 3(image dims - hwc)
        ]
        num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
        for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
            ret.extend([
                tfv1.placeholder(tf.int32, (None, None, None, num_anchors),  # N x H x W x NumAnchors
                               'anchor_labels_lvl{}'.format(k + 2)),
                tfv1.placeholder(tf.float32, (None, None, None, num_anchors, 4),  # N x H x W x NumAnchors x 4
                               'anchor_boxes_lvl{}'.format(k + 2))])
        ret.extend([
            tfv1.placeholder(tf.float32, (None, None, 4), 'gt_boxes'),  # N x MaxNumGTs x 4
            tfv1.placeholder(tf.int64, (None, None), 'gt_labels'),  # all > 0        # N x MaxNumGTs
            tfv1.placeholder(tf.int32, (None,), 'orig_gt_counts')  # N
        ])

        if cfg.MODE_MASK:
            ret.append(
                    tfv1.placeholder(tf.uint8, (None, None, None, None), 'gt_masks')  # N x MaxNumGTs x H x W
            )

        return ret

    def fpn_features(self, image, seed_gen):
        c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS, seed_gen)

        if cfg.TRAIN.BACKBONE_NCHW and not cfg.TRAIN.FPN_NCHW:
            c2345 = [nchw_to_nhwc_transform(c) for c in c2345]
        elif not cfg.TRAIN.BACKBONE_NCHW and cfg.TRAIN.FPN_NCHW:
            c2345 = [nhwc_to_nchw_transform(c) for c in c2345]

        p23456 = fpn_model('fpn', c2345, seed_gen)
        
        if cfg.TRAIN.FPN_NCHW and not cfg.TRAIN.RPN_NCHW:
            p23456 = [nchw_to_nhwc_transform(p) for p in p23456]
        elif not cfg.TRAIN.FPN_NCHW and cfg.TRAIN.RPN_NCHW:
            p23456 = [nhwc_to_nchw_transform(p) for p in p23456]

        return p23456


    def rpn(self, image_shape2d, p_features, anchor_inputs, orig_image_dims, seed_gen):
        """
        The RPN part of the graph that generate the RPN proposal and losses

        Args:
            image_shape2d:  H_image x W_image
            p_features: A List of 5 FPN feature maps, i.e. level P23456
            anchor_inputs: dict, contains all anchor input information
            orig_image_dims: BS x 3
        Returns:
            proposal_rois: BS X top K region proposals X 4
            losses: scalar, sum of the label loss and box loss
        """
        assert len(cfg.RPN.ANCHOR_SIZES) == len(cfg.FPN.ANCHOR_STRIDES)
        
        batch_size = tf.shape(orig_image_dims)[0]
        orig_image_shape2d = orig_image_dims[: ,:2]
    
        all_anchors_fpn = get_all_anchors_fpn()

        multilevel_label_preds = []
        multilevel_box_preds = []

        for p_i in p_features:
            # label_preds: BS x H_feaure x W_feature x NA, 
            # box_preds: BS  x H_feature x W_feature X (NA * 4)
            label_preds, box_preds = rpn_head('rpn', p_i, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS), seed_gen=seed_gen)
            multilevel_label_preds.append(label_preds)
            multilevel_box_preds.append(box_preds)
            
        #  proposal_rois: BS X Top_K X 4 , proposal_scores: [ BS X Top_K X 1 ]
        proposal_rois = generate_fpn_proposals(image_shape2d,
                                                    all_anchors_fpn,
                                                    multilevel_label_preds,
                                                    multilevel_box_preds,
                                                    orig_image_shape2d,
                                                    batch_size)
        if self.training:

            multilevel_anchor_labels = [anchor_inputs['anchor_labels_lvl{}'.format(i + 2)] for i in range(len(all_anchors_fpn))]
            multilevel_anchor_boxes = [anchor_inputs['anchor_boxes_lvl{}'.format(i + 2)] for i in range(len(all_anchors_fpn))]

            multilevel_box_preds_reshaped = []
            for box_preds in multilevel_box_preds:
                shp = tf.shape(box_preds)  # BS x H_feature x W_feature X (NA * 4) 
                box_preds = tf.reshape(box_preds, tf.stack([shp[0], shp[1], shp[2], -1, 4]))  # BS x H_feature x W_feature x NA x 4
                multilevel_box_preds_reshaped.append(box_preds)

            multilevel_rpn_gt = [ RPNGroundTruth(all_anchors_fpn[j], multilevel_anchor_labels[j], multilevel_anchor_boxes[j])  for j in range(len(p_features)) ]
            total_label_loss, total_box_loss = batch_rpn_losses(multilevel_rpn_gt, multilevel_label_preds, multilevel_box_preds_reshaped, orig_image_shape2d) 

            with tf.name_scope('rpn_losses'):
                label_loss = tf.math.truediv(total_label_loss, tf.cast(batch_size, dtype=tf.float32), name='label_loss')
                box_loss = tf.math.truediv(total_box_loss, tf.cast(batch_size, dtype=tf.float32), name='box_loss')
                add_moving_summary(label_loss, box_loss)

                #label_loss = print_runtime_tensor("rpn_losses/label_loss", label_loss)
                #box_loss = print_runtime_tensor("rpn_losses/box_loss", box_loss)

                losses = [label_loss, box_loss]


        else:
            losses = []

        return proposal_rois, losses

    def roi_heads(self, p_features, proposal_rois, ground_truth, inputs, seed_gen):
        """
        Implement the RoI Align and construct the RoI head (box and mask branches) of the graph

        Args:
            p_features: ([tf.Tensor]): A list of 5 FPN feature level P23456
            proposal_rois (tf.Tensor): BS X Num_rois X 4 (x1, y1, x2, y2)
            ground_truth: list of 'gt_boxes', 'gt_labels', 'gt_masks' from input
            inputs: dict, contains all input information
        Returns: all_losses: a list contains box loss and mask loss
        """

        image_shape2d = inputs['orig_image_dims'][: ,:2] # BS x 2
        assert len(p_features) == 5, "Features have to be P23456!"
        gt_boxes, gt_labels, *_ = ground_truth

        prepadding_gt_counts = inputs['orig_gt_counts']

        if self.training:

            # Sample the proposal_rois to make the foreground(fg) box and background(bg) boxes
            # ratio close to configuration. 
            proposal_rois, proposal_labels_gt, proposal_boxes_gt, proposal_fg_gt_indices = sample_fast_rcnn_targets(
                    proposal_rois, # BS X Num_rois X 4 (x1, y1, x2, y2)
                    gt_boxes, # BS X Num_gt_boxes X 4 (x1, y1, x2, y2)
                    gt_labels, # BS x Num_gt_boxes
                    prepadding_gt_counts, # BS
                    seed_gen=seed_gen) 

        p2_5features = p_features[:4] # p2, p3, p4, p5 features

        if cfg.TRAIN.RPN_NCHW:
            p2_5features = [ nchw_to_nhwc_transform(p) for p in p2_5features]

        proposal_rois_y1x1y2x2 = permute_boxes_coords(proposal_rois) # BS X Num_rois X 4 (y1, x1, y2, x2)
        # For Fast R-CNN
        roi_level_ids, roi_features_fastrcnn = roi_features(p2_5features, proposal_rois_y1x1y2x2, 7) #  BS X Num_boxes x H_roi_box x W_roi_box x NumChannel
        with tf.name_scope(name="multilevel_roi_align"):
            roi_level_summary(roi_level_ids)
            
        rff_shape = tf.shape(roi_features_fastrcnn)
        roi_features_fastrcnn = tf.reshape(roi_features_fastrcnn, [-1, rff_shape[2], rff_shape[3], rff_shape[4]])
        fastrcnn_head_func = getattr(boxclass_head, cfg.FPN.BOXCLASS_HEAD_FUNC)
        fastrcnn_head_feature = fastrcnn_head_func('fastrcnn', roi_features_fastrcnn, seed_gen=seed_gen) # Num_sampled_boxes x Num_features

        # fastrcnn_label_preds: Num_sampled_boxes x Num_classes ,fastrcnn_box_preds: Num_sampled_boxes x Num_classes x 4
        fastrcnn_labels_pred, fastrcnn_boxes_pred = boxclass_outputs('fastrcnn/outputs', fastrcnn_head_feature, cfg.DATA.NUM_CLASS, seed_gen=seed_gen)

        fastrcnn_labels_pred = tf.reshape(fastrcnn_labels_pred, [rff_shape[0], rff_shape[1], cfg.DATA.NUM_CLASS] )
        fastrcnn_boxes_pred  = tf.reshape(fastrcnn_boxes_pred, [rff_shape[0], rff_shape[1], cfg.DATA.NUM_CLASS, 4])

        regression_weights = tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32)
        fastrcnn_head = BoxClassHead(fastrcnn_boxes_pred,
                                     fastrcnn_labels_pred,
                                     regression_weights,
                                     proposal_rois)
        if self.training:
            # only calculate the losses for boxes if there is an object (foreground boxes)
            fastrcnn_head.add_training_info(proposal_boxes_gt, proposal_labels_gt)
            all_losses = fastrcnn_head.losses()

            if cfg.MODE_MASK:
                gt_masks = ground_truth[2]
                maskrcnn_head_func = getattr(mask_head, cfg.FPN.MRCNN_HEAD_FUNC)
                
                all_fg_gt_masks = []
                all_fg_gt_labels = []
                all_fg_mask_preds = []
                all_fg_roi_level_ids = []

                for i in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
                    image_gt_count = prepadding_gt_counts[i] # 1-D Num_gt_boxes_current_image
                    image_proposal_gt_labels = proposal_labels_gt[i]
                    image_proposal_fg_indices = tf.reshape(tf.where(image_proposal_gt_labels > 0), [-1])
                    
                    image_proposal_rois_y1x1y2x2 = proposal_rois_y1x1y2x2[i]
                    image_fg_proposal_rois_y1x1y2x2 = tf.gather(image_proposal_rois_y1x1y2x2, image_proposal_fg_indices)
                     
                    image_fg_proposal_rois_y1x1y2x2 = tf.expand_dims(image_fg_proposal_rois_y1x1y2x2, axis=0)
                    image_p2_5features = [ tf.expand_dims(pf[i], axis=0) for pf in p2_5features ]
                    image_fg_roi_level_ids, image_fg_roi_features_maskrcnn = roi_features(image_p2_5features, 
                                                                  image_fg_proposal_rois_y1x1y2x2, 
                                                                  14) # Num_boxes x  H_roi_mask x W_roi_mask x NumChannel
                
                    all_fg_roi_level_ids.extend(image_fg_roi_level_ids)
                        
                    image_fg_roi_features_maskrcnn = tf.squeeze(image_fg_roi_features_maskrcnn, axis=0)

                    if cfg.TRAIN.MASK_NCHW:
                        image_fg_roi_features_maskrcnn = nhwc_to_nchw_transform(image_fg_roi_features_maskrcnn)

                    image_fg_mask_preds = maskrcnn_head_func('maskrcnn', 
                                                             image_fg_roi_features_maskrcnn, 
                                                             cfg.DATA.NUM_CATEGORY, 
                                                             seed_gen=seed_gen)   # Num_boxes x num_category x (H_roi_mask*2) x (W_roi_mask*2
                    all_fg_mask_preds.append(image_fg_mask_preds)
                   
                    image_proposal_fg_gt_labels = tf.gather(image_proposal_gt_labels, image_proposal_fg_indices) # 1-D Num_fg_boxes_current_image
                    
                    image_gt_masks = gt_masks[i, :image_gt_count, :, :] # Num_gt_boxes_current_image x H_gtmask x W_gtmask
                    image_gt_masks = tf.expand_dims(image_gt_masks, axis=3) # Num_gt_boxes_current_image x H_gtmask x W_gtmask X 1
                    image_proposal_fg_gt_indices = proposal_fg_gt_indices[i]

                    image_fg_proposal_rois_y1x1y2x2 = tf.squeeze(image_fg_proposal_rois_y1x1y2x2, axis=0)
                    image_proposal_fg_gt_masks = crop_and_resize(image_gt_masks, 
                                                                image_fg_proposal_rois_y1x1y2x2, 
                                                                image_proposal_fg_gt_indices, 
                                                                28, 
                                                                image_shape2d[i], 
                                                                pad_border=False) # Num_fg_boxes_current_image x  (H_roi_mask*2) x (W_roi_mask*2) x 1
                   
                    all_fg_gt_labels.append(image_proposal_fg_gt_labels)
                    all_fg_gt_masks.append(image_proposal_fg_gt_masks)

                with tf.name_scope(name="multilevel_roi_align_mask"):
                    roi_level_ids = []
                    for i in range(4):
                        roi_level_ids.append(tf.concat(all_fg_roi_level_ids[i::4], axis=0))
                    roi_level_summary(roi_level_ids)
                  
                fg_gt_masks = tf.concat(all_fg_gt_masks, axis=0) # Num_fg_boxes x (H_roi_mask*2) x (W_roi_mask*2) X 1
                fg_gt_labels = tf.concat(all_fg_gt_labels, axis=0) # 1-D Num_fg_boxes

                fg_gt_masks = tf.squeeze(fg_gt_masks, 3, 'fg_gt_masks') # Num_fg_boxes x (H_roi_mask*2) x (W_roi_mask*2)
                fg_mask_preds = tf.concat(all_fg_mask_preds, axis=0) # Num_fg_boxes x num_category x (H_roi_mask*2) x (W_roi_mask*2)
                mask_loss = maskrcnn_loss(fg_mask_preds, fg_gt_labels, fg_gt_masks)
                #mask_loss = print_runtime_tensor("mask_loss", mask_loss)

                all_losses.append(mask_loss)
            return all_losses
        else:
            decoded_boxes = fastrcnn_head.decoded_output_boxes_batch() # BS X N x #class x 4 
            decoded_boxes = clip_boxes_batch(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes') # BS X N X (#class) X 4 (x1y1x2y2)
            label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores') # BS X N x #class scores
    
            final_labels_list = []
            final_boxes_list = []
            final_scores_list = []
            batch_indicies_list = []
            final_masks_list = []

            maskrcnn_head_func = getattr(mask_head, cfg.FPN.MRCNN_HEAD_FUNC)
            for i in range(cfg.TEST.BATCH_SIZE_PER_GPU):
                image_decoded_boxes = decoded_boxes[i]
                image_label_scores = label_scores[i]
                image_final_boxes, image_final_scores, image_final_labels, image_box_ids = boxclass_predictions(image_decoded_boxes, image_label_scores)
                image_batch_ids = tf.tile([i], [tf.size(image_box_ids)])

                final_boxes_list.append(image_final_boxes)
                final_scores_list.append(image_final_scores)
                final_labels_list.append(image_final_labels)
                batch_indicies_list.append(image_batch_ids)

                if cfg.MODE_MASK:
                    image_final_boxes_y1x1y2x2 = permute_boxes_coords(image_final_boxes)
                    image_final_boxes_y1x1y2x2 = tf.expand_dims(image_final_boxes_y1x1y2x2, axis=0)
                    image_p2_5features = [ tf.expand_dims(pf[i], axis=0) for pf in p2_5features ]

                    _, image_roi_features_maskrcnn = roi_features(image_p2_5features, 
                                                            image_final_boxes_y1x1y2x2, 
                                                            14) # 1 X Num_boxes x  H_roi_mask x W_roi_mask x NumChannel

                    irfm_shape = tf.shape(image_roi_features_maskrcnn)
                    image_roi_features_maskrcnn = tf.reshape(image_roi_features_maskrcnn, [-1, irfm_shape[2], irfm_shape[3], irfm_shape[4]])
                    if cfg.TRAIN.MASK_NCHW:
                        image_roi_features_maskrcnn = nhwc_to_nchw_transform(image_roi_features_maskrcnn)
                    
                    image_mask_logits = maskrcnn_head_func('maskrcnn', image_roi_features_maskrcnn, cfg.DATA.NUM_CATEGORY, seed_gen=seed_gen)   # N x #cat x 28 x 28
                    
                    image_label_indices = tf.stack([tf.range(tf.size(image_final_labels)), tf.cast(image_final_labels, tf.int32) - 1], axis=1)
                    image_mask_logits = tf.gather_nd(image_mask_logits, image_label_indices)   # #resultx28x28

                    image_mask = tf.sigmoid(image_mask_logits)
                    final_masks_list.append(image_mask)

           
            with tf.name_scope(name="output"):
                batch_indices = tf.identity(tf.concat(batch_indicies_list, 0), name="batch_indices")
                final_boxes = tf.identity(tf.concat(final_boxes_list, 0), name="boxes")
                final_scores = tf.identity(tf.concat(final_scores_list, 0), name="scores")
                final_labels = tf.identity(tf.concat(final_labels_list, 0), name="labels")

                if cfg.MODE_MASK:
                    final_masks = tf.identity(tf.concat(final_masks_list, 0), name="masks")

            return []

