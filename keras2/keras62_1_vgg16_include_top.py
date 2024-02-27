import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
tf.random.set_seed(777)
np.random.seed(777)
# print(tf.__version__)

from keras.applications import VGG16

# model = VGG16()
# 기본값 : include_top=True, input_shape(224,224,3)

    # =================================================================
    # Total params: 138,357,544  /  1억3835만7544
    # Trainable params: 138,357,544
    # Non-trainable params: 0
    # _________________________________________________________________
model = VGG16(
              weights='imagenet', 
              include_top=False,    # 기본값 True
              input_shape=(32, 32, 3)
              )
# False로 주면 FullyConnectedLayer(Dense)는 안씀. input_shape를 조정가능
# Dense = FullyConnectedLayer
    # =================================================================
    # Total params: 14,714,688
    # Trainable params: 14,714,688
    # Non-trainable params: 0
    # _________________________________________________________________
model.summary()

#################### include_top = Falsea ####################
#1. FC layer 날림
#2. input_shape 내가 하고싶은대로

