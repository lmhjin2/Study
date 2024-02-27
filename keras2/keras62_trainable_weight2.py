import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
# tf.random.set_seed(777)
# np.random.seed(777)
# print(tf.__version__)

# 1 data
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2 model
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
                     # kernel == weights
print(model.weights) # weights의 초기값은 랜덤. bias는 0.
print("="*50)
print(model.trainable_weights)
print("="*50)

print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 6

########################################
model.trainable=False # ★★★
########################################

print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 0
# 이건 전이학습에 씀

