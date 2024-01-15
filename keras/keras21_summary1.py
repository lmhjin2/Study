import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

#1
x = np.array([1,2,3])
y = np.array([1,2,3])

#2
model = Sequential()
model.add(Dense(5, input_shape=(1,)))   # vector 하나 들어감
model.add(Dense(4))
model.add(Dense(2,))
model.add(Dense(1))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 5)                 10
#  dense_1 (Dense)             (None, 4)                 24
#  dense_2 (Dense)             (None, 2)                 10
#  dense_3 (Dense)             (None, 1)                 3
# =================================================================
# Total params: 47 (188.00 Byte)
# Trainable params: 47 (188.00 Byte)
# Non-trainable params: 0 (0.00 Byte)




