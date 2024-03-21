# 01 ~ 11 까지 cnn으로 만들어서 성능비교

# from sklearn.datasets import load_breast_cancer
# datasets = load_breast_cancer()
# x = datasets.data
# y = datasets.target

# print(x.shape, y.shape) # (569, 30) (569,)
# # (569, 30) -> (569,15,2,1) or (569,10,3,1) or (569,3,5,2)

# restore_best_weights 와
# save_best_only 에 대한 고찰


import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston

#1
datasets = load_boston()

x = datasets.data
y = datasets.target

# print(x.shape)  # (506, 13)
# print(y.shape)  # (506,)




random_state_value = 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = random_state_value )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# # test_csv = scaler.transform(test_csv)

print( x_train.shape, x_test.shape)
x_train = x_train.reshape(-1,13,1,1)
x_test = x_test.reshape(-1,13,1,1)
print(x_train.shape, x_test.shape)
# (404, 13, 1, 1) (102, 13, 1, 1)

print(y_train.shape, y_test.shape)



#2
model = Sequential()
model.add(Conv2D(15, (2,1), input_shape=(13,1,1), padding='same',
                 activation='sigmoid'))
model.add(Conv2D(13, (2,1), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(13, (2,1), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(82, activation='relu'))
model.add(Dense(46, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

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

es = EarlyStopping(monitor='val_loss', mode = 'auto',
                   patience = 50, verbose = 1, restore_best_weights = True )
# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose = 1, save_best_only = True,
#     filepath=filepath)

model.compile(loss='mae', optimizer='adam')
import time as tm
start_time = tm.time()
hist = model.fit(x_train, y_train,
          validation_split = 0.2,
          epochs = 500, batch_size = 32 )
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

# CNN
# 로스 : 2.4913604259490967
# R2 스코어 : 0.8828381360238552
# run time: 22.77









