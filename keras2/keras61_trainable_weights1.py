import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
tf.random.set_seed(777)
np.random.seed(777)
# print(tf.__version__)

# 1 data
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2 model
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

########################################
# model.trainable=False # ★★★
model.trainable=True # ★★★s 기본값
########################################
print(model.weights)
# model.summary()

#3. compile, train
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,batch_size=1, epochs=1000, verbose=0)

#4 predict
y_pred = model.predict(x)
print(y_pred)