import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
tf.random.set_seed(777)
np.random.seed(777)
# print(tf.__version__)

from keras.applications import VGG16

# model = VGG16()
# =================================================================
# Total params: 138,357,544  /  1억3835만7544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________
model = VGG16(weights='imagenet', 
              include_top=False, 
              input_shape=(32, 32, 3))
model.summary()


