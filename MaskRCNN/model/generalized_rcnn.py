# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import six
assert six.PY3, "FasterRCNN requires Python 3!"
import tensorflow as tf
import math

from tensorpack import *
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary

from model import mask_head, boxclass_head
from model.backbone import image_preprocess, resnet_fpn_backbone
from config import config as cfg
from data import get_all_anchors_fpn
from model_box import RPNAnchors, clip_boxes_batch, crop_and_resize
from model.fpn import fpn_model, multilevel_roi_align
from model.boxclass_head import boxclass_predictions, boxclass_outputs, BoxClassHead
from model.biased_sampler import sample_fast_rcnn_targets
from model.mask_head import maskrcnn_loss
from model.rpn import rpn_head, multilevel_rpn_losses, generate_fpn_proposals, generate_fpn_proposals_topk_per_image
from utils.randomnness import SeedGenerator


class GradientClipOptimizer(tf.train.Optimizer):
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
        all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in old_grads])
        clipped_grads, _ = tf.clip_by_global_norm(old_grads, self.clip_norm,
                                         use_norm=tf.cond(
                                         all_are_finite,
                                         lambda: tf.global_norm(old_grads),
                                         lambda: tf.constant(self.clip_norm, dtype=tf.float32)), name='clip_by_global_norm')
        gradvars = list(zip(clipped_grads, v))
        #gradvars[0] = (print_runtime_tensor_loose_branch('NORM ', norm, prefix=f'rank{hvd.rank()}', trigger_tensor=gradvars[0][0]), gradvars[0][1])
        return self.opt.apply_gradients(gradvars, global_step, name)

    def get_slot(self, *args, **kwargs):
        return self.opt.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self.opt.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self.opt.variables(*args, **kwargs)


class DetectionModel(ModelDesc):
    def __init__(self, fp16):
        self.fp16 = fp16

    def preprocess(self, image):
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    @property
    def training(self):
        return get_current_tower_context().is_training

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)


        opt = tf.train.MomentumOptimizer(lr, 0.9)
        if cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
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
        inputs = dict(zip(self.input_names, inputs))

        image = self.preprocess(inputs['images'])     # NCHW

        seed_gen = SeedGenerator(cfg.TRAIN.SEED)

        features = self.backbone(image, seed_gen)

        anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}
        proposal_boxes, rpn_losses = self.rpn(image, features, anchor_inputs, inputs['orig_image_dims'], seed_gen)  # inputs?

        targets = [inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]
        head_losses = self.roi_heads(image, features, proposal_boxes, targets, inputs, seed_gen)

        if self.training:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            total_cost = tf.add_n(rpn_losses + head_losses + [wd_cost], 'total_cost')
            #total_cost = print_runtime_tensor('COST ', total_cost, prefix=f'rank{hvd.rank()}')
            add_moving_summary(total_cost, wd_cost)
            return total_cost


class ResNetFPNModel(DetectionModel):
    def __init__(self, fp16):
        super(ResNetFPNModel, self).__init__(fp16)

    def inputs(self):

        ret = [
            tf.placeholder(tf.string, (None,), 'filenames'), # N length vector of filenames
            tf.placeholder(tf.float32, (None, None, None, 3), 'images'),  # N x H x W x C
            tf.placeholder(tf.int32, (None, 3), 'orig_image_dims')  # N x 3(image dims - hwc)
        ]
        num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
        for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
            ret.extend([
                tf.placeholder(tf.int32, (None, None, None, num_anchors),  # N x H x W x NumAnchors
                               'anchor_labels_lvl{}'.format(k + 2)),
                tf.placeholder(tf.float32, (None, None, None, num_anchors, 4),  # N x H x W x NumAnchors x 4
                               'anchor_boxes_lvl{}'.format(k + 2))])
        ret.extend([
            tf.placeholder(tf.float32, (None, None, 4), 'gt_boxes'),  # N x MaxNumGTs x 4
            tf.placeholder(tf.int64, (None, None), 'gt_labels'),  # all > 0        # N x MaxNumGTs
            tf.placeholder(tf.int32, (None,), 'orig_gt_counts')  # N
        ])

        if cfg.MODE_MASK:
            ret.append(
                    tf.placeholder(tf.uint8, (None, None, None, None), 'gt_masks')  # N x MaxNumGTs x H x W
            )

        return ret


    def backbone(self, image, seed_gen):
        c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS, seed_gen=seed_gen, fp16=self.fp16)
        print("c2345", c2345)
        p23456 = fpn_model('fpn', c2345, seed_gen=seed_gen, fp16=self.fp16)
        return p23456


    def rpn(self, image, features, inputs, orig_image_dims, seed_gen):
        """
        The RPN part of the graph that generate the RPN proposal and losses

        Args:
            image: BS x NumChannel x H_image x W_image
            features: ([tf.Tensor]): A list of 5 FPN feature maps, i.e. level P23456, each with BS x NumChannel x H_feature x W_feature
            inputs: dict, contains all input information
            orig_image_dims: BS x 3
        Returns:
            proposal_boxes: top K region proposals, K x 5
            losses: scalar, sum of the label loss and box loss
        """
        assert len(cfg.RPN.ANCHOR_SIZES) == len(cfg.FPN.ANCHOR_STRIDES)

        image_shape2d = orig_image_dims[: ,:2]

        all_anchors_fpn = get_all_anchors_fpn()

        rpn_outputs = []
        for pi in features:
            # label_logits: BS x H_feaure x W_feature x NA, box_logits: BS x (NA * 4) x H_feature x W_feature
            label_logits, box_logits = rpn_head('rpn', pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS), seed_gen=seed_gen, fp16=self.fp16)
            rpn_outputs.append((label_logits, box_logits))

        multilevel_label_logits = [k[0] for k in rpn_outputs] # Num_level * [BS x H_feature x W_feature x NA]
        multilevel_box_logits = [k[1] for k in rpn_outputs] # Num_level * [BS x (NA * 4) x H_feature x W_feature]

        # proposal_boxes: K x 5, proposal_scores: 1-D K
        if cfg.RPN.TOPK_PER_IMAGE:
            proposal_boxes, proposal_scores = generate_fpn_proposals_topk_per_image(all_anchors_fpn,
                                                                                    multilevel_box_logits,
                                                                                    multilevel_label_logits,
                                                                                    image_shape2d,
                                                                                    cfg.TRAIN.BATCH_SIZE_PER_GPU)
        else:

            proposal_boxes, proposal_scores = generate_fpn_proposals(all_anchors_fpn,
                                                                     multilevel_box_logits,
                                                                     multilevel_label_logits,
                                                                     image_shape2d,
                                                                     cfg.TRAIN.BATCH_SIZE_PER_GPU)
        if self.training:

            multilevel_anchor_labels = [inputs['anchor_labels_lvl{}'.format(i + 2)] for i in range(len(all_anchors_fpn))]
            multilevel_anchor_boxes = [inputs['anchor_boxes_lvl{}'.format(i + 2)] for i in range(len(all_anchors_fpn))]

            multilevel_box_logits_reshaped = []
            for box_logits in multilevel_box_logits:
                shp = tf.shape(box_logits)  # BS x (NA * 4) x H_feature x W_feature
                box_logits_t = tf.transpose(box_logits, [0, 2, 3, 1])  # BS x H_feature x W_feature x (NA * 4)
                box_logits_t = tf.reshape(box_logits_t, tf.stack([shp[0], shp[2], shp[3], -1, 4]))  # BS x H_feature x W_feature x NA x 4
                multilevel_box_logits_reshaped.append(box_logits_t)

            rpn_losses  = []
            for i in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):
                orig_image_hw = orig_image_dims[i, :2]
                si_all_anchors_fpn = get_all_anchors_fpn()
                si_multilevel_box_logits = [box_logits[i] for box_logits in multilevel_box_logits_reshaped] # [H_feature x W_feature x NA x 4] * Num_levels
                si_multilevel_label_logits = [label_logits[i] for label_logits in multilevel_label_logits] # [H_feature x W_feature x NA] * Num_levels
                si_multilevel_anchor_labels = [anchor_labels[i] for anchor_labels in multilevel_anchor_labels]
                si_multilevel_anchors_boxes = [anchor_boxes[i] for anchor_boxes in multilevel_anchor_boxes]

                si_multilevel_anchors = [RPNAnchors(si_all_anchors_fpn[j],
                                                    si_multilevel_anchor_labels[j],
                                                    si_multilevel_anchors_boxes[j])
                                         for j in range(len(features))]

                # Given the original image dims, find what size each layer of the FPN feature map would be (follow FPN padding logic)
                mult = float \
                    (cfg.FPN.RESOLUTION_REQUIREMENT)  # the image is padded so that it is a multiple of this (32 with default config).
                orig_image_hw_after_fpn_padding = tf.ceil(tf.cast(orig_image_hw, tf.float32) / mult) * mult
                featuremap_dims_per_level = []
                for lvl, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
                    featuremap_dims_float = orig_image_hw_after_fpn_padding / float(stride)
                    featuremap_dims_per_level.append \
                        (tf.cast(tf.math.floor(featuremap_dims_float + 0.5), tf.int32))  # Fix bankers rounding

                si_multilevel_anchors_narrowed = [anchors.narrow_to_featuremap_dims(dims) for anchors, dims in zip(si_multilevel_anchors, featuremap_dims_per_level)]
                si_multilevel_box_logits_narrowed = [box_logits[:dims[0], :dims[1] ,: ,:] for box_logits, dims in zip(si_multilevel_box_logits, featuremap_dims_per_level)]
                si_multilevel_label_logits_narrowed = [label_logits[:dims[0], :dims[1] ,:] for label_logits, dims in zip(si_multilevel_label_logits, featuremap_dims_per_level)]

                si_losses = multilevel_rpn_losses(si_multilevel_anchors_narrowed,
                                                  si_multilevel_label_logits_narrowed,
                                                  si_multilevel_box_logits_narrowed)
                rpn_losses.extend(si_losses)


            with tf.name_scope('rpn_losses'):
                total_label_loss = tf.truediv(tf.add_n(rpn_losses[::2]), tf.cast(cfg.TRAIN.BATCH_SIZE_PER_GPU, dtype=tf.float32), name='label_loss')
                total_box_loss = tf.truediv(tf.add_n(rpn_losses[1::2]), tf.cast(cfg.TRAIN.BATCH_SIZE_PER_GPU, dtype=tf.float32), name='box_loss')
                add_moving_summary(total_label_loss, total_box_loss)
                losses = [total_label_loss, total_box_loss]

        else:
            losses = []

        return proposal_boxes, losses

    def roi_heads(self, image, features, proposal_boxes, targets, inputs, seed_gen):
        """
        Implement the RoI Align and construct the RoI head (box and mask branches) of the graph

        Args:
            image: BS x NumChannel x H_image x W_image
            features: ([tf.Tensor]): A list of 5 FPN feature level P23456, each with BS X NumChannel X H_feature X W_feature
            proposal_boxes(tf.Tensor): K x 5 boxes
            targets: list of 'gt_boxes', 'gt_labels', 'gt_masks' from input
            inputs: dict, contains all input information
        Returns: all_losses: a list contains box loss and mask loss
        """

        image_shape2d = inputs['orig_image_dims'][: ,:2] # BS x 2

        assert len(features) == 5, "Features have to be P23456!"
        gt_boxes, gt_labels, *_ = targets

        prepadding_gt_counts = inputs['orig_gt_counts']

        if self.training:
            input_proposal_boxes = proposal_boxes # K x 5
            input_gt_boxes = gt_boxes # BS x Num_gt_boxes x 4
            input_gt_labels = gt_labels # BS x Num_gt_boxes

            # Sample the input_proposal_boxes to make the foreground(fg) box and background(bg) boxes
            # ratio close to configuration. proposal_boxes: Num_sampled_boxs x 5, proposal_labels: 1-D Num_sampled_boxes
            # proposal_gt_id_for_each_fg contains indices for matching GT of each foreground box.
            proposal_boxes, proposal_labels, proposal_gt_id_for_each_fg = sample_fast_rcnn_targets(
                    input_proposal_boxes,
                    input_gt_boxes,
                    input_gt_labels,
                    prepadding_gt_counts,
                    batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, seed_gen=seed_gen)

        # For the box/class branch
        roi_feature_fastrcnn = multilevel_roi_align(features[:4], proposal_boxes, 7) # Num_sampled_boxes x NumChannel x H_roi_box x W_roi_box
        fastrcnn_head_func = getattr(boxclass_head, cfg.FPN.BOXCLASS_HEAD_FUNC)
        head_feature = fastrcnn_head_func('fastrcnn', roi_feature_fastrcnn, seed_gen=seed_gen, fp16=self.fp16) # Num_sampled_boxes x Num_features
        # fastrcnn_label_logits: Num_sampled_boxes x Num_classes ,fastrcnn_box_logits: Num_sampled_boxes x Num_classes x 4
        fastrcnn_label_logits, fastrcnn_box_logits = boxclass_outputs('fastrcnn/outputs', head_feature, cfg.DATA.NUM_CLASS, seed_gen=seed_gen)
        regression_weights = tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32)

        fastrcnn_head = BoxClassHead(fastrcnn_box_logits,
                                     fastrcnn_label_logits,
                                     regression_weights,
                                     prepadding_gt_counts,
                                     proposal_boxes)
        if self.training:
            # only calculate the losses for boxes if there is an object (foreground boxes)
            proposal_fg_inds = tf.reshape(tf.where(proposal_labels > 0), [-1])
            proposal_fg_boxes = tf.gather(proposal_boxes, proposal_fg_inds)
            proposal_fg_labels = tf.gather(proposal_labels, proposal_fg_inds)

            fastrcnn_head.add_training_info(input_gt_boxes,
                                            proposal_labels,
                                            proposal_fg_inds,
                                            proposal_fg_boxes,
                                            proposal_fg_labels,
                                            proposal_gt_id_for_each_fg)

            all_losses = fastrcnn_head.losses(cfg.TRAIN.BATCH_SIZE_PER_GPU)

            if cfg.MODE_MASK:
                gt_masks = targets[2]

                maskrcnn_head_func = getattr(mask_head, cfg.FPN.MRCNN_HEAD_FUNC)

                # For the mask branch. roi_feature_maskrcnn: Num_fg_boxes x NumChannel x H_roi_mask x W_roi_mask
                roi_feature_maskrcnn = multilevel_roi_align(
                        features[:4], proposal_fg_boxes, 14,
                        name_scope='multilevel_roi_align_mask')

                mask_logits = maskrcnn_head_func(
                        'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY, seed_gen=seed_gen, fp16=self.fp16)   # Num_fg_boxes x num_category x (H_roi_mask*2) x (W_roi_mask*2)
                per_image_target_masks_for_fg = []
                per_image_fg_labels = []
                for i in range(cfg.TRAIN.BATCH_SIZE_PER_GPU):

                    single_image_gt_count = prepadding_gt_counts[i] # 1-D Num_gt_boxes_current_image
                    single_image_gt_masks = gt_masks[i, :single_image_gt_count, :, :] # Num_gt_boxes_current_image x H_gtmask x W_gtmask
                    single_image_fg_indices = tf.squeeze(tf.where(tf.equal(proposal_fg_boxes[:, 0], i)), axis=1) # 1-D Num_fg_boxes_current_image
                    single_image_fg_boxes = tf.gather(proposal_fg_boxes, single_image_fg_indices)[:, 1:] # Num_fg_boxes_current_image x 4
                    single_image_fg_labels = tf.gather(proposal_fg_labels, single_image_fg_indices) # 1-D Num_fg_boxes_current_image
                    single_image_fg_inds_wrt_gt = proposal_gt_id_for_each_fg[i] # 1-D Num_fg_boxes_current_image

                    assert isinstance(single_image_fg_inds_wrt_gt, tf.Tensor)

                    single_image_gt_masks = tf.expand_dims(single_image_gt_masks, axis=1) # Num_gt_boxes_current_image x 1 x H_gtmask x W_gtmask
                    # single_image_target_masks_for_fg: Num_fg_boxes_current_image x 1 x (H_roi_mask*2) x (W_roi_mask*2)
                    single_image_target_masks_for_fg = crop_and_resize(single_image_gt_masks,
                                                                       single_image_fg_boxes,
                                                                       single_image_fg_inds_wrt_gt,
                                                                       28,
                                                                       image_shape2d[i],
                                                                       pad_border=False,
                                                                       verbose_batch_index=i)
                    per_image_fg_labels.append(single_image_fg_labels)
                    per_image_target_masks_for_fg.append(single_image_target_masks_for_fg)

                target_masks_for_fg = tf.concat(per_image_target_masks_for_fg, axis=0) # Num_fg_boxes x 1 x (H_roi_mask*2) x (W_roi_mask*2)

                proposal_fg_labels = tf.concat(per_image_fg_labels, axis=0) # 1-D Num_fg_boxes

                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets') # Num_fg_boxes x (H_roi_mask*2) x (W_roi_mask*2)
                mask_loss = maskrcnn_loss(mask_logits, proposal_fg_labels, target_masks_for_fg)

                all_losses.append(mask_loss)
            return all_losses
        else:

            decoded_boxes, batch_ids = fastrcnn_head.decoded_output_boxes_batch()
            decoded_boxes = clip_boxes_batch(decoded_boxes, image_shape2d, tf.cast(batch_ids, dtype=tf.int32), name='fastrcnn_all_boxes')
            label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')

            final_boxes, final_scores, final_labels, box_ids = boxclass_predictions(decoded_boxes, label_scores, name_scope='output')
            batch_indices = tf.gather(proposal_boxes[: ,0], box_ids, name='output/batch_indices')

            if cfg.MODE_MASK:

                batch_ind_boxes = tf.concat((tf.expand_dims(batch_indices, 1), final_boxes), axis=1)

                roi_feature_maskrcnn = multilevel_roi_align(features[:4], batch_ind_boxes, 14)
                maskrcnn_head_func = getattr(mask_head, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                        'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY, seed_gen=seed_gen, fp16=self.fp16)   # #fg x #cat x 28 x 28
                indices = tf.stack([tf.range(tf.size(final_labels)), tf.cast(final_labels, tf.int32) - 1], axis=1)
                final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx28x28
                tf.sigmoid(final_mask_logits, name='output/masks')

            return []
