import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import time as tm
from sklearn.datasets import fetch_california_housing

#1. DATA
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
# print(x.shape)  # (20640, 8)
# print(y.shape)  # (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 3)

#2. MODEL
model = Sequential()
model.add(Dense(16, input_dim = 8))
model.add(Dense(26))
model.add(Dense(17))
model.add(Dense(8))
model.add(Dense(1))



#3. COMPILE, TRAIN
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 500,
          validation_split = 0.15, verbose = 2)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('loss:', loss)
print('r2 : ', r2)


# loss: 0.9514561295509338
# r2 :  0.2806226306096028

