# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import numpy as np
import os
import six
import pprint

from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu

__all__ = ['config', 'finalize_configs']


class AttrDict():

    _freezed = False
    """ Avoid accidental creation of new hierarchies. """

    def __getattr__(self, name):
        if self._freezed:
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name, value):
        if self._freezed and name not in self.__dict__:
            raise AttributeError(
                "Config was freezed! Unknown config: {}".format(name))
        super().__setattr__(name, value)

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1, width=100, compact=True)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items() if not k.startswith('_')}

    def update_args(self, args):
        """Update from command line args. """
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)

    def freeze(self, freezed=True):
        self._freezed = freezed
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.freeze(freezed)

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()


config = AttrDict()
_C = config     # short alias to avoid coding

# mode flags ---------------------
_C.TRAINER = 'replicated'  # options: 'horovod', 'replicated'
_C.MODE_MASK = True        # FasterRCNN or MaskRCNN
_C.MODE_FPN = True

# dataset -----------------------
_C.DATA.BASEDIR = '/path/to/your/DATA/DIR'
# All TRAIN dataset will be concatenated for training.
_C.DATA.TRAIN = ['train2017']   # i.e. trainval35k, AKA train2017
# Each VAL dataset will be evaluated separately (instead of concatenated)
_C.DATA.VAL = ('val2017', )  # AKA val2017
# This two config will be populated later by the dataset loader:
_C.DATA.NUM_CATEGORY = 0  # without the background class (e.g., 80 for COCO)
_C.DATA.CLASS_NAMES = []  # NUM_CLASS (NUM_CATEGORY+1) strings, the first is "BG".

# basemodel ----------------------
_C.BACKBONE.WEIGHTS = ''   # /path/to/weights.npz
_C.BACKBONE.RESNET_NUM_BLOCKS = [3, 4, 6, 3]     # for resnet50
# RESNET_NUM_BLOCKS = [3, 4, 23, 3]    # for resnet101
_C.BACKBONE.FREEZE_AFFINE = False   # do not train affine parameters inside norm layers
_C.BACKBONE.NORM = 'FreezeBN'  # options: FreezeBN, SyncBN, GN, None
_C.BACKBONE.FREEZE_AT = 2  # options: 0, 1, 2

# Use a base model with TF-preferred padding mode,
# which may pad more pixels on right/bottom than top/left.
# See https://github.com/tensorflow/tensorflow/issues/18213
# In tensorpack model zoo, ResNet models with TF_PAD_MODE=False are marked with "-AlignPadding".
# All other models under `ResNet/` in the model zoo are using TF_PAD_MODE=True.
# Using either one should probably give the same performance.
# We use the "AlignPadding" one just to be consistent with caffe2.
_C.BACKBONE.TF_PAD_MODE = False
_C.BACKBONE.STRIDE_1X1 = False  # True for MSRA models

# schedule -----------------------
_C.TRAIN.NUM_GPUS = None         # by default, will be set from code
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BASE_LR = 1.25e-2  / 8.0 # defined for total batch size=1. It will be adjusted automatically using linear scaling.
_C.TRAIN.WARMUP_STEPS = 1000   # in terms of iterations. This is not affected by #GPUs
_C.TRAIN.WARMUP_INIT_LR = 1e-2 * 0.33 / 8.0 # defined for total batch size=1. It will be adjusted automatically
_C.TRAIN.STARTING_EPOCH = 1  # the first epoch to start with, useful to continue a training

_C.TRAIN.LR_EPOCH_SCHEDULE = [(8, 0.1), (10, 0.01), (12, None)] # "1x" schedule in detectron
_C.TRAIN.EVAL_PERIOD = 1 # period (epochs) to run evaluation
_C.TRAIN.BATCH_SIZE_PER_GPU = 1
_C.TRAIN.SEED = 1234
_C.TRAIN.GRADIENT_CLIP = 0 # set non-zero value to enable gradient clip, 0.36 is recommended for 32x4
_C.TRAIN.BACKBONE_NCHW = False
_C.TRAIN.FPN_NCHW = False
_C.TRAIN.RPN_NCHW = False
_C.TRAIN.MASK_NCHW = False
_C.TRAIN.SHOULD_STOP = False # use stop the training early (for async eval)

# preprocessing --------------------
# Alternative old (worse & faster) setting: 600
_C.PREPROC.TRAIN_SHORT_EDGE_SIZE = [800, 800]  # [min, max] to sample from
_C.PREPROC.TEST_SHORT_EDGE_SIZE = 800
_C.PREPROC.MAX_SIZE = 1333
# mean and std in RGB order.
# Un-scaled version: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_C.PREPROC.PIXEL_MEAN = [123.675, 116.28, 103.53]
_C.PREPROC.PIXEL_STD = [58.395, 57.12, 57.375]
_C.PREPROC.PREDEFINED_PADDING = False
_C.PREPROC.PADDING_SHAPES = [(800, 1000), (800, 1200), (800, 1350)]    # only add landscape shapes in decreasing h/w aspect ratio order - the corresponding portrait shape will be automatically created

# anchors -------------------------
_C.RPN.ANCHOR_STRIDE = 16
_C.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)   # sqrtarea of the anchor box
_C.RPN.ANCHOR_RATIOS = (0.5, 1., 2.)
_C.RPN.POSITIVE_ANCHOR_THRESH = 0.7
_C.RPN.NEGATIVE_ANCHOR_THRESH = 0.3

# rpn training -------------------------
_C.RPN.FG_RATIO = 0.5  # fg ratio among selected RPN anchors
_C.RPN.BATCH_PER_IM = 256  # total (across FPN levels) number of anchors that are marked valid
_C.RPN.MIN_SIZE = 0.1
_C.RPN.PROPOSAL_NMS_THRESH = 0.7
# Anchors which overlap with a crowd box (IOA larger than threshold) will be ignored.
# Setting this to a value larger than 1.0 will disable the feature.
# It is disabled by default because Detectron does not do this.
_C.RPN.CROWD_OVERLAP_THRESH = 9.99

# RPN proposal selection -------------------------------
_C.RPN.TRAIN_PER_LEVEL_PRE_NMS_TOPK = 4000
_C.RPN.TRAIN_PER_LEVEL_POST_NMS_TOPK = 2000
_C.RPN.TRAIN_IMAGE_POST_NMS_TOPK = 2000
_C.RPN.TEST_PER_LEVEL_PRE_NMS_TOPK = 2000
_C.RPN.TEST_PER_LEVEL_POST_NMS_TOPK = 1000
_C.RPN.TEST_IMAGE_POST_NMS_TOPK = 1000
_C.RPN.UNQUANTIZED_ANCHOR = True # From tensorpack https://github.com/tensorpack/tensorpack/commit/141ab53cc37dce728802803747584fc0fb82863b

# fastrcnn training ---------------------
_C.FRCNN.BATCH_PER_IM = 512
_C.FRCNN.BBOX_REG_WEIGHTS = [10., 10., 5., 5.]  # Better but non-standard setting: [20, 20, 10, 10]
_C.FRCNN.FG_THRESH = 0.5
_C.FRCNN.FG_RATIO = 0.25  # fg ratio in a ROI batch

# FPN -------------------------
_C.FPN.ANCHOR_STRIDES = (4, 8, 16, 32, 64)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
_C.FPN.PROPOSAL_MODE = 'Level'  # Must be set to 'Level'
_C.FPN.NUM_CHANNEL = 256
_C.FPN.NORM = 'None'  # 'None', 'GN'
# The head option is only used in FPN. For C4 models, the head is C5
_C.FPN.BOXCLASS_HEAD_FUNC = 'boxclass_2fc_head'
# choices: boxclass_2fc_head, boxclass_4conv1fc_{,gn_}head
_C.FPN.BOXCLASS_CONV_HEAD_DIM = 256
_C.FPN.BOXCLASS_FC_HEAD_DIM = 1024
_C.FPN.MRCNN_HEAD_FUNC = 'maskrcnn_up4conv_head'   # choices: maskrcnn_up4conv_{,gn_}head

# Mask-RCNN
_C.MRCNN.HEAD_DIM = 256

# testing -----------------------
_C.TEST.FRCNN_NMS_THRESH = 0.5

# Smaller threshold value gives significantly better mAP. But we use 0.05 for consistency with Detectron.
# mAP with 1e-4 threshold can be found at https://github.com/tensorpack/tensorpack/commit/26321ae58120af2568bdbf2269f32aa708d425a8#diff-61085c48abee915b584027e1085e1043  # noqa
_C.TEST.RESULT_SCORE_THRESH = 0.05
_C.TEST.RESULT_SCORE_THRESH_VIS = 0.3   # only visualize confident results
_C.TEST.RESULTS_PER_IM = 100
_C.TEST.BOX_TARGET = 0.377
_C.TEST.MASK_TARGET = 0.339
_C.TEST.BATCH_SIZE_PER_GPU = 1

_C.freeze()  # avoid typo / wrong config keys


def finalize_configs(is_training):
    """
    Run some sanity checks, and populate some configs from others
    """
    _C.freeze(False)  # populate new keys now
    _C.DATA.NUM_CLASS = _C.DATA.NUM_CATEGORY + 1  # +1 background
    _C.DATA.BASEDIR = os.path.expanduser(_C.DATA.BASEDIR)
    if isinstance(_C.DATA.VAL, six.string_types):  # support single string (the typical case) as well
        _C.DATA.VAL = (_C.DATA.VAL, )

    assert _C.BACKBONE.NORM in ['FreezeBN', 'SyncBN', 'GN', 'None'], _C.BACKBONE.NORM
    if _C.BACKBONE.NORM != 'FreezeBN':
        assert not _C.BACKBONE.FREEZE_AFFINE
    assert _C.BACKBONE.FREEZE_AT in [0, 1, 2]

    _C.RPN.NUM_ANCHOR = len(_C.RPN.ANCHOR_SIZES) * len(_C.RPN.ANCHOR_RATIOS)
    assert len(_C.FPN.ANCHOR_STRIDES) == len(_C.RPN.ANCHOR_SIZES)

    # image size into the backbone has to be multiple of this number
    _C.FPN.RESOLUTION_REQUIREMENT = _C.FPN.ANCHOR_STRIDES[3]  # [3] because we build FPN with features r2,r3,r4,r5

    if _C.MODE_FPN:
        size_mult = _C.FPN.RESOLUTION_REQUIREMENT * 1.
        _C.PREPROC.MAX_SIZE = np.ceil(_C.PREPROC.MAX_SIZE / size_mult) * size_mult
        assert _C.FPN.PROPOSAL_MODE in ['Level']
        assert _C.FPN.BOXCLASS_HEAD_FUNC.endswith('_head')
        assert _C.FPN.MRCNN_HEAD_FUNC.endswith('_head')
        assert _C.FPN.NORM in ['None', 'GN']
    else:
        raise NotImplementedError("MODE_FPN=False is not implemented")

    if is_training:
        train_scales = _C.PREPROC.TRAIN_SHORT_EDGE_SIZE
        if isinstance(train_scales, (list, tuple)) and train_scales[1] - train_scales[0] > 100:
            # don't autotune if augmentation is on
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
        os.environ['TF_AUTOTUNE_THRESHOLD'] = '1'
        assert _C.TRAINER in ['horovod', 'replicated'], _C.TRAINER

        # setup NUM_GPUS
        if _C.TRAINER == 'horovod':
            import horovod.tensorflow as hvd
            ngpu = hvd.size()

            if ngpu == hvd.local_size():
                logger.warn("It's not recommended to use horovod for single-machine training. "
                            "Replicated trainer is more stable and has the same efficiency.")
        else:
            assert 'OMPI_COMM_WORLD_SIZE' not in os.environ
            ngpu = get_num_gpu()
        assert ngpu % 8 == 0 or 8 % ngpu == 0, "Can only train with 1,2,4 or >=8 GPUs, but found {} GPUs".format(ngpu)
    else:
        # autotune is too slow for inference
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
        ngpu = get_num_gpu()

    assert ngpu > 0, "Has to run with GPU!"
    if _C.TRAIN.NUM_GPUS is None:
        _C.TRAIN.NUM_GPUS = ngpu
    else:
        if _C.TRAINER == 'horovod':
            assert _C.TRAIN.NUM_GPUS == ngpu
        else:
            assert _C.TRAIN.NUM_GPUS <= ngpu

    _C.freeze()
    logger.info("Config: ------------------------------------------\n" + str(_C))
