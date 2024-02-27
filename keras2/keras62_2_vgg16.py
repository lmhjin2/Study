import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
tf.random.set_seed(777)
np.random.seed(777)
# print(tf.__version__)
from keras.datasets import cifar10
from keras.applications import VGG16

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg16.trainable = False  # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

model.fit(x_train, y_train)

model.evaluate(x_test,y_test)
