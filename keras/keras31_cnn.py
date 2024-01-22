# Convolutional Neural Networks

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential

model = Sequential() # (2,2) 를 커널 사이즈라고 함
model.add(Conv2D(10, (2, 2), input_shape = (10,10,1)))    # input = (n,3)
model.add(Dense(5))
model.add(Dense(1))

