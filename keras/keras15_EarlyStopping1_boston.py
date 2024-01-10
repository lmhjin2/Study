import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
import time as tm

from sklearn.datasets import load_boston
## 아래는 09_1 카피.
datasets = load_boston()
x = datasets.data
y = datasets.target
# print(x.shape)  # (506, 13)
# print(y.shape)  # (506,)

# 실습
# train_size 0.7 이상 0.9 이하
# R2 0.62 이상 / 0.8 이상



x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 1 )

model = Sequential()
model.add(Dense(26, input_dim = 13))
model.add(Dense(52))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(13))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# start_time = round(tm.time(), 2)
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',
                   mode = 'min',
                   patience = 50,
                   verbose = 1,
                   restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs = 200, batch_size = 1,
          validation_split = 0.2, verbose = 1,
          callbacks=[es])   # list로 묶은건 친구들이 있다는 뜻.

# end_time = round(tm.time(), 2)

loss = model.evaluate(x_test, y_test)   # 생각해 보니 이놈도 실행됨. 내가 잘못 알고 있었음
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

# def RMSE(aaa, bbb):
#     return np.sqrt(mean_squared_error(aaa, bbb))
# rmse = RMSE(y_test, y_predict)

# def RMSLE(aaa, bbb):
#     return np.sqrt(mean_squared_log_error(aaa, bbb))
# rmsle = RMSLE(y_test, y_predict)

# print("로스 :", loss) # print("예측값 :", y_predict)
print("R2 스코어 :", r2)
# print("걸린시간 : ", round(end_time - start_time, 2), "초")
print('MSE: ', loss)    # 로스 : 랑 같은값임.
# print('RMSE:', rmse)
# print('RMSLE:', rmsle)

# print('=========================== hist ===================================')
# print(hist)
# # <keras.src.callbacks.History object at 0x000001C2F2B38E90>
# print('===================== hist.history ==================================')
# print(hist.history)
# print('=========================== loss ====================================')
# print(hist.history['loss']) # 로스값이 들어있음.
# print('======================== val loss ===================================')
print(hist.history['val_loss'])
# print('=====================================================================')

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color = 'red', label = 'loss', marker = '.')
# plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')
plt.legend(loc = 'upper right') # loc = location # upper - lower / right - left
plt.title('보스턴 loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()      

import warnings
warnings.filterwarnings('ignore')

# patience
# R2 스코어 : 0.7274254961623279
# MSE:  26.93782615661621

# restore_best_weights = True
# R2 스코어 : 0.739571757205811
# MSE:  25.737443923950195