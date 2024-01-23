import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
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
x = x.reshape(569, 6,5,1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state = 1 )

#2 
model = Sequential()
model.add(Conv2D(33, (2,2), input_shape=(6,5,1), 
                 padding='same', activation='sigmoid'))
model.add(Conv2D(12, (3,3), padding='same', activation='relu'))
model.add(Conv2D(36, (5,4), padding='same', activation='relu'))
model.add(Conv2D(21, (4,4), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(123, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(1, activation='relu'))

#3
# import datetime

# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")   # 월일_시분

# path1 = "c:/_data/_save/MCP/k28/06/"
# filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
# filepath = "".join([path1, 'k28_', date, '_', filename])

model.compile(loss = 'binary_crossentropy', optimizer='adam',
              metrics=['accuracy','mse','mae'])
es = EarlyStopping(monitor = 'val_loss', mode = 'auto',
                   patience = 50, verbose = 1,
                   restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
#                       verbose = 1, save_best_only=True,
#     filepath=filepath)

import time as tm
start_time = tm.time()

hist = model.fit(x_train, y_train, epochs = 1000,
                 batch_size = 210, validation_split = 0.13,
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
# loss: [0.20226573944091797, 0.9534883499145508, 0.046718303114175797, 0.06178978830575943]
# r2: 0.8080396679605841


# scaler = MinMaxScaler()
# loss: [0.08510842174291611, 0.9651162624359131, 0.02643170952796936, 0.05430173873901367]
# acc:  0.9651162790697675
# r2: 0.8894293325067316

# scaler = StandardScaler()
# loss: [0.15428347885608673, 0.9418604373931885, 0.044408977031707764, 0.0717775747179985]
# acc:  0.9418604651162791
# r2: 0.8142258012453736

# scaler = MaxAbsScaler()
# loss: [0.11676334589719772, 0.9534883499145508, 0.03654831647872925, 0.06537999957799911]
# acc:  0.9534883720930233
# r2: 0.8471089766129697

# scaler = RobustScaler()
# loss: [0.07344205677509308, 0.9651162624359131, 0.02196618542075157, 0.054660264402627945]
# acc:  0.9651162790697675
# r2: 0.9081097883593787

# RobustScaler

# CPU
# 20.91 초

# GPU
# 22.98 초

# CNN
# loss: [6.028769016265869, 0.604651153087616, 13.220781326293945, 3.6093218326568604]
# acc:  0.0
# r2: -54.3059383496941

