import sys
import numpy as np
import tensorflow as tf
from tensorpack import *

sys.path.append('/mask-rcnn-tensorflow/MaskRCNN')

NCHW=False
STRIDE_1X1=False
BACKBONE_NCHW=False
NORM=False
MODE_FPN = True
TF_PAD_MODE = False
MODEL_WEIGHTS = '/data/pretrained-models/ImageNet-R50-AlignPadding.npz'

@layer_register(log_shape=True)
def GroupNorm(x, group=32, gamma_initializer=tf.constant_initializer(1.)):
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')

def maybe_reverse_pad(topleft, bottomright):
    if TF_PAD_MODE:
        return [topleft, bottomright]
    return [bottomright, topleft]

def resnet_shortcut(l, n_out, stride, seed_gen, activation=tf.identity):
    n_in = l.shape[1] if BACKBONE_NCHW else l.shape[-1]
    if n_in != n_out:   # change dimension when channel is not the same
        # TF's SAME mode output ceil(x/stride), which is NOT what we want when x is odd and stride is 2
        # In FPN mode, the images are pre-padded already.
        if not MODE_FPN and stride == 2:
            l = l[:, :, :-1, :-1] if BACKBONE_NCHW else l[:, :-1, :-1, :]
        return Conv2D('convshortcut', l, n_out, 1,
                      strides=stride, activation=activation, seed=seed_gen.next())
    else:
        return l

def get_norm(zero_init=False):
    if NORM == 'None':
        return lambda x: x
    if NORM == 'GN':
        Norm = GroupNorm
        layer_name = 'gn'
    else:
        Norm = BatchNorm
        layer_name = 'bn'

    def norm(x):
        dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x = Norm(layer_name, x, gamma_initializer=tf.zeros_initializer() if zero_init else None)
        return tf.cast(x, dtype)

class SeedGenerator:
    def __init__(self, seed):
        self.seed = seed
        self.counters = dict()

    def next(self, key='default'):
        if self.seed == None:
            return None

        if key not in self.counters:
            self.counters[key] = self.seed
            return self.counters[key]
        else:
            self.counters[key] += 1
            return self.counters[key]

def resnet_group(name, l, block_func, features, count, stride, seed_gen):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1, seed_gen)
    return l

def resnet_bottleneck(l, ch_out, stride, seed_gen):
    shortcut = l
    if STRIDE_1X1:
        if stride == 2:
            l = l[:, :, :-1, :-1] if NCHW else l[:, :-1, :-1, :]
        l = Conv2D('conv1', l, ch_out, 1, strides=stride, seed=seed_gen.next())
        l = Conv2D('conv2', l, ch_out, 3, strides=1, seed=seed_gen.next())
    else:
        l = Conv2D('conv1', l, ch_out, 1, strides=1, seed=seed_gen.next())
        if stride == 2:
            if BACKBONE_NCHW:
                l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(0, 1), maybe_reverse_pad(0, 1)])
            else:
                l = tf.pad(l, [[0, 0], maybe_reverse_pad(0, 1), maybe_reverse_pad(0, 1), [0, 0]])
            l = Conv2D('conv2', l, ch_out, 3, strides=2, padding='VALID', seed=seed_gen.next())
        else:
            l = Conv2D('conv2', l, ch_out, 3, strides=stride, seed=seed_gen.next())
    if not NORM:
        l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_norm(zero_init=True), seed=seed_gen.next())
    else:
        l = Conv2D('conv3', l, ch_out * 4, 1, activation=tf.identity,
                   kernel_initializer=tf.constant_initializer(), seed=seed_gen.next())
    ret = l + resnet_shortcut(shortcut, ch_out * 4, stride, seed_gen=seed_gen, activation=get_norm(zero_init=False))
    return tf.nn.relu(ret, name='output')

# define a small version of the backbone model
def resnet_fpn_backbone(image, num_blocks, seed_gen, fp16=False):
    c2 = resnet_group('group0', image, resnet_bottleneck, 64, num_blocks[0], 1, seed_gen=seed_gen)
    return c2

result = resnet_fpn_backbone(tf.random_normal((10, 3, 64, 64)), [3, 4, 6, 3], SeedGenerator(1234))

session_init = get_model_loader(MODEL_WEIGHTS)
sess = tf.Session()
session_init._run_init(sess)
init = tf.global_variables_initializer()
sess.run(init)
test = sess.run(result)
print(test.mean())
