# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import tensorflow as tf
from contextlib import suppress

def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    norm = "norm" in name.lower() or "bn" in name.lower()
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer,
                      regularizer=regularizer if not norm else None,
                      trainable=trainable,
                      *args, **kwargs)

    # print(name, "trainable={} dtype={} storage_dtype={} id={} reuse={}".format(trainable, dtype, storage_dtype, id(variable), kwargs['reuse']))

    if norm:
        return variable

    if trainable and dtype != tf.float32:
        # print(name, "fp16_cast")
        cast_name = name + '/fp16_cast'
        try:
            cast_variable = tf.get_default_graph().get_tensor_by_name(
                cast_name + ':0'
            )
        except KeyError:
            cast_variable = tf.cast(variable, dtype, name=cast_name)
        cast_variable._ref = variable._ref
        variable = cast_variable
    return variable


def mixed_precision_scope(mixed=True, *args, **kwargs):
    if not mixed:
        return suppress()

    return tf.variable_scope(name_or_scope=tf.get_variable_scope(),
                             custom_getter=float32_variable_storage_getter,
                             reuse=tf.AUTO_REUSE, *args, **kwargs)

