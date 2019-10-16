import sys
import numpy as np
import tensorflow as tf
from tensorpack import *

image = tf.random_normal((10, 3, 64, 64))
result = Conv2D('conv1', image, 64, 1, strides=1, seed=1234)

x = {'conv1/W:0': np.random.normal(size=(1,1,64,64)).astype(np.float32)}
np.savez("weights.npz", **x)

session_init = get_model_loader("weights.npz")
sess = tf.Session()
session_init._run_init(sess)
init = tf.global_variables_initializer()
sess.run(init)

test = sess.run(result)

print(test.mean())
