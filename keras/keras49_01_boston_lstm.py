# restore_best_weights 와
# save_best_only 에 대한 고찰


import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
# from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

#1
datasets = load_boston()

x = datasets.data
y = datasets.target

random_state_value = 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = random_state_value )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)
x_train = x_train.reshape(-1,13,1)
x_test = x_test.reshape(-1,13,1)
# print(x_train.shape, x_test.shape)
# (404, 13) (102, 13)

#2
model = Sequential()
model.add(LSTM(26, input_shape = (13,1) , activation='swish'))    # 364
model.add(Dropout(0.2))
model.add(Dense(52, activation='swish'))                    # 1404
model.add(Dropout(0.3))
model.add(Dense(25, activation='swish'))                    # 1325
model.add(Dense(50, activation='swish'))                    # 1300
model.add(Dropout(0.3))
model.add(Dense(13, activation='swish'))                    # 663
model.add(Dropout(0.5))
model.add(Dense(1))                     # 14
# model.summary()   Dropout 있으나 없으나 똑같이 나옴
# dropout 기본값 = 0.5
# evaluate predict 에선 적용 안됨. 할 필요도 x 
# 과적합 방지에 좋음

#3
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

date = datetime.datetime.now()
# print(date) # 2024-01-17 10:53:07.663370
# <class 'datetime.datetime'>

date = date.strftime("%m%d_%H%M")   # 월 / 일 _ 시 : 분
# print(date) # 0117_1059
# <class 'str'>

# path = "c:/_data/_save/MCP/k28/01/"
# filename = "{epoch:04d}-{val_loss:.4f}.hdf5"    # 예) 1234-0.3333.hdf5
# # 04d 는 정수 네자리 수 까지(뭐의 약자인진 모름), .4f는 소수 넷째자리까지 표시하라는뜻 (float)
# filepath = "".join([path, 'k28_', date, '_', filename])
# # '../_data/_save/Mcp/k25_0117_1059_0101-0.3333.hdf5'

# es = EarlyStopping(monitor='val_loss', mode = 'auto',
#                    patience = 10, verbose = 1, restore_best_weights = True )
# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose = 1, save_best_only = True,
#     filepath=filepath)

model.compile(loss='mae', optimizer='adam')
import time as tm
start_time = tm.time()
hist = model.fit(x_train, y_train,
          validation_split = 0.2,
          epochs = 1000, batch_size = 32 )
end_time = tm.time()
run_time = round(end_time - start_time, 2)
# ModelCheckpoint

#4
print('============================== 1. 기본 출력 ====================================')

loss = model.evaluate(x_test, y_test, verbose = 0)
print("로스 :", loss)

y_predict = model.predict(x_test, verbose = 0)
r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)

print("run time:", run_time)

# print('=================================================================================')
# print(hist.history)
# print('=================================================================================')
import warnings
warnings.filterwarnings('ignore')

# restore_best_weights 와
# save_best_only 에 대한 고찰

# True, True    18개 파일. loss순으로 나열됨
# True, False   epoch 마다 하나씩 만듬.
# False, True   loss순으로 저장되긴하는데 당장 모델에 좋은값이 안박힘
# False, False  epoch 마다 하나씩 만드는데 당장 모델에 좋은값이 안박힘

# 그냥 True 두개 박아두고 쓰자.



# CPU
# 25.53 초

# GPU
# 36.85 초

# 로스 : 3.1463305950164795
# R2 스코어 : 0.8170148862959209
# run time: 169.71