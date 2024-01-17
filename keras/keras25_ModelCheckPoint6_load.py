# 5번에서 저장된 파일을 불러서 


import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
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

#2
model = Sequential()
model.add(Dense(26, input_dim = 13))    # 364
model.add(Dense(52))                    # 1404
model.add(Dense(25))                    # 1325
model.add(Dense(50))                    # 1300
model.add(Dense(13))                    # 663
model.add(Dense(1))                     # 14
# # model.summary()

#3
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

# date = datetime.datetime.now()
# # print(date) # 2024-01-17 10:53:07.663370
# # <class 'datetime.datetime'>

# date = date.strftime("%m%d_%H%M")   # 월 / 일 _ 시 : 분
# # print(date) # 0117_1059
# # <class 'str'>

# path = "c:/_data/_save/MCP/k25/FF/"
# filename = "{epoch:04d}-{val_loss:.4f}.hdf5"    # 예) 1234-0.3333.hdf5
# # 04d 는 정수 네자리 수 까지(뭐의 약자인진 모름), .4f는 소수 넷째자리까지 표시하라는뜻 (float)
# filepath = "".join([path, 'k25_', date, '_', filename])
# # '../_data/_save/Mcp/k25_0117_1059_0101-0.3333.hdf5'

# es = EarlyStopping(monitor='val_loss', mode = 'auto',
#                    patience = 10, verbose = 1, restore_best_weights = True )
# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose = 1, save_best_only = True,
#     filepath=filepath)

# model.compile(loss='mae', optimizer='adam')
# hist = model.fit(x_train, y_train,
#           callbacks = [es, mcp], validation_split = 0.2,
#           epochs = 1000, batch_size = 32 )
# model.save('c:/_data/_save/keras25_3_save_model.h5')

# ModelCheckpoint\
model = load_model('c:/_data/_save/MCP/k25/TT/k25_0117_1143_0032-3.1084.hdf5')

#4
print('============================== 1. 기본 출력 ====================================')

loss = model.evaluate(x_test, y_test, verbose = 0)
print("로스 :", loss)

y_predict = model.predict(x_test, verbose = 0)
r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)


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
