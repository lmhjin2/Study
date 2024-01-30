# LSTM에서는 출력, 입력, 삭제 게이트라는 3개의 게이트가 존재했습니다.
# 반면, GRU에서는 업데이트 게이트와 리셋 게이트 두 가지 게이트만이 존재합니다.
# GRU는 LSTM보다 학습 속도가 빠르다고 알려져있지만 여러 평가에서 GRU는 LSTM과 비슷한 성능을 보인다고 알려져 있습니다.

# cell state 하나 때문에 lstm이 네배 gru가 세배인데 gru가 요즘 연산식이 바뀐듯함 + 10%정도
# lstm보다 조금 빠르지만 성능이조금 떨어진다는 평가. 가끔은 더 좋음

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, Bidirectional, LSTM, GRU
from keras.callbacks import EarlyStopping

#1
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4, 5, 6, 7, 8, 9, 10])

# print(x.shape, y.shape) # (7, 3) (7,)
x = x.reshape(7, 3, 1)
# print(x.shape, y.shape) # (7, 3, 1) (7,)

#2
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape = (3,1)))   # timesteps, features
# model.add(Bidirectional(SimpleRNN(units = 10), input_shape = (3,1)))   # timesteps, features
# 3-D tensor with shape (batch_size, timesteps, features).
# model.add(Bidirectional(LSTM(units = 10), input_shape = (3,1)))   # timesteps, features
model.add(GRU( units = 10, input_shape = (3,1)))   # timesteps, features
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary() # 10 : 240 /  20 : 880 / 5 : 70

# #3
# es = EarlyStopping(monitor='loss', mode = 'auto',
#                    patience = 200, verbose = 1,
#                    restore_best_weights = True)
# model.compile(loss = 'mse', optimizer='adam')
# model.fit(x, y, epochs = 10000, callbacks=[es])

# #4
# results = model.evaluate(x, y)
# y_pred = np.array([8,9,10]).reshape(1,3,1)
# y_pred = model.predict(y_pred)

# print('loss', results)
# print('[8,9,10]의 결과', y_pred)
# # [[10.xxxxx]] output = 2차원

