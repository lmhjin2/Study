# 원핫하고, 모델 완성해서 돌려라!!!
# acc = 0.98 이상

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Reshape, Conv1D, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import time as tm

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28* 28)
x_test = x_test.reshape(10000, 28* 28)

scaler = MinMaxScaler() # 정규화 한다하면 스케일링한거임.

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)/255 == (60000, 28, 28, 1)/255
# print(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/255 == (10000, 28, 28, 1)/ 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2
model = Sequential()
model.add(Dense(9, input_shape = (28, 28, 1)))   
model.add(Conv2D(10, (3,3)))   
model.add(Reshape(target_shape = (26*26, 10)))
model.add(Conv1D(15, 4))             
model.add(LSTM(8, return_sequences=True))
model.add(Conv1D(14, 2))
model.add(Dense(units=8))
model.add(Dense(7, input_shape = (8,)))
model.add(Flatten())
model.add(Dense(6))
model.add(Dense(10, activation='softmax'))

model.summary()

# #3
# es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1,
#                    patience = 30 , restore_best_weights=True )
# # mcp = ModelCheckpoint(monitor='val_accuracy', mode = 'auto', verbose=1)


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# start_time = tm.time()
# model.fit(x_train, y_train, validation_split = 0.15, 
#           batch_size = 500, verbose = 1, epochs = 10 , callbacks=[es])
# end_time=tm.time()
# run_time=round(end_time - start_time, 2)

# #4 
# y_predict = model.predict(x_test)
# results = model.evaluate(x_test, y_test)

# y_train = np.argmax(y_train, axis=1)
# y_test = np.argmax(y_test, axis=1)
# y_predict = np.argmax(y_predict, axis=1)

# acc = accuracy_score(y_test, y_predict)

# print('run time', run_time)
# print('loss', results[0])
# print('acc', results[1], acc)

# # loss: 0.0266 - accuracy: 0.9930

# # run time 8.43
# # loss 0.0852874368429184
# # acc 0.9767000079154968 0.9767

# # run time 49.35
# # loss 0.10515447705984116
# # acc 0.9857000112533569 0.9857


