import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1 data
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2 model
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 28,28,1])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 10])

w1 = tf.compat.v1.get_variable('w1', shape = [2, 2, 1, 64])
                                      # 커널사이즈, 컬러(채널), 필터(아웃풋)
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID') # 4차원이라서 [1,1,1,1], 중간 두개가 stride고 앞뒤의 1은 그냥 shape 맞춰주는놈임. 두칸씩 하려면 [1,2,2,1]. 2번이 가로 3번이 세로
# model.add(Conv2D(64, kernel_size=(2,2), input_shape=(28,28,1), stride=(1,1)))
