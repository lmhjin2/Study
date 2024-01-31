import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)   # (569, 30)
# print(datasets.feature_names)
x = datasets.data
y = datasets.target
# numpy
# print(np.unique(y)) # [ 0 1 ]  

# print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))

# pandas 넷다똑같음
# print(pd.DataFrame(y).value_counts())
# print(pd.Series(y).value_counts())
# print(pd.Series.value_counts(y))
# print(pd.value_counts(y))

# 1    357
# 0    212


# print(x.shape, y.shape) # (569, 30) (569,)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state = 1 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1,30,1)
x_test = x_test.reshape(-1,30,1)

print(x_train.shape, x_test.shape)

#2 
model = Sequential()
model.add(Conv1D(110, 10, input_shape = (30,1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(80))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dense(1, activation = 'sigmoid'))

#3
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   # 월일_시분

path1 = "c:/_data/_save/MCP/k28/06/"
filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
filepath = "".join([path1, 'k28_', date, '_', filename])

model.compile(loss = 'binary_crossentropy', optimizer='adam',
              metrics=['accuracy','mse','mae'])
# es = EarlyStopping(monitor = 'val_loss', mode = 'auto',
#                    patience = 100, verbose = 1,
#                    restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
#                       verbose = 1, save_best_only=True,
#     filepath=filepath)

import time as tm
start_time = tm.time()

hist = model.fit(x_train, y_train, epochs = 1000,
                 batch_size = 105, validation_split = 0.13,
                 verbose = 2)
end_time = tm.time()
run_time = round(end_time - start_time, 2)
# loss: 0.09776605665683746
# r2: 0.5982901288073281

#4 
loss = model.evaluate(x_test, y_test)   # x_test를 넣어서 predict한 값을 y_test 와 비교.
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
# print("======================================================================================")
# print(y_test)
# print("======================================================================================")
# print(y_predict)
# print("======================================================================================")

def ACC(y_test, y_predict):
    return accuracy_score(y_test, y_predict)

acc = ACC(y_test, np.round(y_predict))


print('loss:', loss)
print('acc: ', acc)
print('r2:', r2)
print("run time:", run_time)
print('breast cancer')

# scaler = MinMaxScaler()
# loss: [0.08510842174291611, 0.9651162624359131, 0.02643170952796936, 0.05430173873901367]
# acc:  0.9651162790697675
# r2: 0.8894293325067316

# scaler = RobustScaler()
# loss: [0.07344205677509308, 0.9651162624359131, 0.02196618542075157, 0.054660264402627945]
# acc:  0.9651162790697675
# r2: 0.9081097883593787

# RobustScaler

# LSTM
# loss: [0.8299967050552368, 0.930232584476471, 0.06865294277667999, 0.07142984122037888]
# acc:  0.9302325581395349
# r2: 0.7128070145699857

# Conv1D
# loss: [0.6073618531227112, 0.9651162624359131, 0.034909069538116455, 0.03558975085616112]
# acc:  0.9651162790697675
# r2: 0.8539663700776812
# run time: 26.83
