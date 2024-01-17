# 09_1 copy
# 세이브 파일 만들기
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

date = datetime.datetime.now()
# print(date) # 2024-01-17 10:53:07.663370
# <class 'datetime.datetime'>

date = date.strftime("%m%d_%H%M")   # 월 / 일 _ 시 : 분
# print(date) # 0117_1059
# <class 'str'>

path = "c:/_data/_save/MCP/k25/"
filename = "{epoch:04d}-{val_loss:.4f}.hdf5"    # 예) 1234-0.3333.hdf5
# 04d 는 정수 네자리 수 까지(뭐의 약자인진 모름), .4f는 소수 넷째자리까지 표시하라는뜻 (float)
filepath = "".join([path, 'k25_', date, '_', filename])
# '../_data/_save/Mcp/k25_0117_1059_0101-0.3333.hdf5'

es = EarlyStopping(monitor='val_loss', mode = 'auto',
                   patience = 100, verbose = 1, restore_best_weights = True )
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose = 1, save_best_only=True,
    filepath=filepath)

model.compile(loss='mae', optimizer='adam')
hist = model.fit(x_train, y_train,
          callbacks = [es, mcp], validation_split = 0.2,
          epochs = 1000, batch_size = 32 )
# model.save('c:/_data/_save/keras25_3_save_model.h5')

# ModelCheckpoint

#4
print('============================== 1. 기본 출력 ====================================')

loss = model.evaluate(x_test, y_test, verbose = 0)
print("로스 :", loss)

y_predict = model.predict(x_test, verbose = 0)
r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)


# print('========================== 2. load_moedel 출력 ====================================')

# model2 = load_model('c:/_data/_save/keras25_3_save_model.h5')

# loss2 = model2.evaluate(x_test, y_test, verbose = 0)
# print("로스 :", loss2)

# y_predict2 = model2.predict(x_test, verbose = 0)
# r2 = r2_score(y_test, y_predict2)
# print("R2 스코어 :", r2)

# print('========================== 3. load_moedel 출력 ====================================')


# model3 = load_model("c:/_data/_save/MCP/keras_25_MCP3.hdf5")

# loss3 = model3.evaluate(x_test, y_test, verbose = 0)
# print("로스 :", loss3)

# y_predict3 = model3.predict(x_test, verbose = 0)
# r2 = r2_score(y_test, y_predict3)
# print("R2 스코어 :", r2)


# print("Random State:", random_state_value)
# print("boston")
# print('=================================================================================')
# print(hist.history)
# print('=================================================================================')
import warnings
warnings.filterwarnings('ignore')

# 0.7845982031606218

# random_state = 1, epochs = 8000, batch_size = 30, loss = 'mse'
# 로스 : 18.694143295288086
# R2 스코어 : 0.796036779999476
# 걸린시간 :  58.3 초

# scaler = MinMaxScaler()
# 로스 : 3.1486480236053467
# R2 스코어 : 0.7835180345040087
# 걸린시간 :  9.9 초

# scaler = StandardScaler()
# 로스 : 3.144235610961914
# R2 스코어 : 0.7939326089286648
# 걸린시간 :  9.63 초

# scaler = MaxAbsScaler()
# 로스 : 3.134629487991333
# R2 스코어 : 0.7852473736299466
# 걸린시간 :  9.71 초

# scaler = RobustScaler()
# 로스 : 3.1448447704315186
# R2 스코어 : 0.7995639930028042
# 걸린시간 :  9.69 초


# RoubustScaler

