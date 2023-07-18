# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf

from tensorpack.models import Conv2D, FixedUnPooling, MaxPooling, layer_register
from tensorpack.tfutils.argscope import argscope

from model.backbone import GroupNorm
from config import config as cfg

@layer_register(log_shape=True)
def fpn_model(features, seed_gen):
    """
    Args:
        features ([tf.Tensor]): ResNet features c2-c5

    Returns:
        [tf.Tensor]: FPN features p2-p6
    """
    assert len(features) == 4, features
    num_channel = cfg.FPN.NUM_CHANNEL

    use_gn = cfg.FPN.NORM == 'GN'

    def upsample2x(name, x):
        dtype_str = 'float32'
        return FixedUnPooling(
            name, x, 2, unpool_mat=np.ones((2, 2), dtype=dtype_str),
            data_format='channels_first' if cfg.TRAIN.FPN_NCHW else 'channels_last')

    with argscope(Conv2D, data_format='channels_first' if cfg.TRAIN.FPN_NCHW else 'channels_last',
                  activation=tf.identity, use_bias=True,
                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1., seed=seed_gen.next())):
        lat_2345 = [Conv2D('lateral_1x1_c{}'.format(i + 2), c, num_channel, 1, seed=seed_gen.next())
                    for i, c in enumerate(features)]
        if use_gn:
            lat_2345 = [GroupNorm('gn_c{}'.format(i + 2), c) for i, c in enumerate(lat_2345)]
        lat_sum_5432 = []
        for idx, lat in enumerate(lat_2345[::-1]):
            if idx == 0:
                lat_sum_5432.append(lat)
            else:
                lat = lat + upsample2x('upsample_lat{}'.format(6 - idx), lat_sum_5432[-1])
                lat_sum_5432.append(lat)
        p2345 = [Conv2D('posthoc_3x3_p{}'.format(i + 2), c, num_channel, 3, seed=seed_gen.next())
                 for i, c in enumerate(lat_sum_5432[::-1])]
        if use_gn:
            p2345 = [GroupNorm('gn_p{}'.format(i + 2), c) for i, c in enumerate(p2345)]
        p6 = MaxPooling('maxpool_p6', p2345[-1], pool_size=1, strides=2, data_format='channels_first' if cfg.TRAIN.FPN_NCHW else 'channels_last', padding='VALID')

        return p2345 + [p6]
