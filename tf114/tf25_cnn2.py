import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1 data
x_train = np.array([[[[1],[2],[3]],
                     [[4],[5],[6]],
                     [[7],[8],[9]]]])
print(x_train.shape)    # (1, 3, 3, 1)

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 3, 3, 1])
w = tf.compat.v1.constant()