import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,mean_absolute_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
plt.rcParams['font.family'],"Malgun Gothic"
plt.rcParams['axes.unicode_minus']=False

#1 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state=0 )

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)
# (17544, 8) (3096, 8)

x_train = x_train.reshape(-1,8,1)
x_test = x_test.reshape(-1,8,1)

#2
model = Sequential()
model.add(LSTM(16, input_shape = (8,1)))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dropout(0.2))
model.add(Dense(8))
model.add(Dense(1))

#3
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   # 월일_시분

path = "c:/_data/_save/MCP/k28/02/"
filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
filepath = "".join([path, 'k28_', date, '_', filename])
model.compile(loss='mse',optimizer='adam',
              metrics = ['mse','msle', 'mae'])
# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience = 300, verbose = 1,
#                    restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose = 1,
#                       save_best_only=True,
#                       filepath=filepath)
import time as tm
start_time = tm.time()
hist = model.fit(x_train, y_train, epochs=1000,
                 batch_size = 301, validation_split=0.2,
                 verbose = 2)
end_time = tm.time()
run_time = round(end_time - start_time, 2)
#4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test,y_predict))

rmse = RMSE(y_test, y_predict)
# rmsle = RMSLE(y_test, y_predict)
# print("음수갯수 :", y[y['count']<0].count())  ## 진짜 중요함 ##

print('loss:', loss)
print('RMSE:', rmse)
# print('RMSLE:', rmsle)
print('R2:',r2)
print("run time:", run_time)
'''
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color = 'red',
         label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], color = 'blue',
         label = 'val_loss', marker = '.')
plt.legend(loc='upper right')
plt.title('캘리포니아 로스')
plt.xlabel('epochs')
plt.ylabel('val_loss')
plt.grid()
plt.show()
'''

# loss: [0.5937129259109497, 0.5937129259109497, 0.05482317879796028, 0.5589895248413086]
# RMSE: 0.7705276989046251
# R2: 0.5519150654481986

# scaler = MinMaxScaler()
# loss: [0.5377083420753479, 0.5377083420753479, 0.052736494690179825, 0.5330697298049927]
# RMSE: 0.733286021331258
# R2: 0.5941826188806718

# scaler = StandardScaler()
# loss: [0.5548639893531799, 0.5548639893531799, 0.05236700922250748, 0.5384661555290222]
# RMSE: 0.7448919710115381
# R2: 0.5812349600575786

# scaler = MaxAbsScaler()
# loss: [0.5390101075172424, 0.5390101075172424, 0.052398569881916046, 0.5343481302261353]
# RMSE: 0.7341730965529952
# R2: 0.5932001692557324

# scaler = RobustScaler()
# loss: [1.26486074924469, 1.26486074924469, 0.05075812339782715, 0.5429605841636658]
# RMSE: 1.1246603976563418
# R2: 0.045388554455149444


# MixMaxScaler


# CPU
# 72.45 초

# GPU
# 89.32 초


# loss: [0.5205868482589722, 0.5205868482589722, 0.05125166475772858, 0.5364072918891907]
# RMSE: 0.7215170672223521
# R2: 0.6071045051844135
# run time: 130.16