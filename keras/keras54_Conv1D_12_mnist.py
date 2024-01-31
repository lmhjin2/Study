import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, GRU, Conv1D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)
# print(x_train)

# x_train = x_train.reshape(60000, 28*28)
# print(x_train.shape[0]) # 60000
# x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_test.shape[0]) # 10000

# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
# print(x_train.shape, x_test.shape)
# (60000, 784) (10000, 784)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

x_train = x_train.reshape(-1,28,28)
x_test = x_test.reshape(-1,28,28)

# print(y_train.shape)


#2
model = Sequential()
model.add(Conv1D(80, 2, input_shape=(28,28), activation='sigmoid'))
model.add(Dense(50, activation='swish'))
model.add(Dense(30, activation='swish'))
model.add(LSTM(20, activation='swish'))
model.add(Dense(20, activation='swish'))
model.add(Dense(80, activation='swish'))
model.add(Dense(50, activation='swish'))
model.add(Dense(10, activation='softmax'))


#3
es = EarlyStopping(monitor = 'val_loss', mode = 'auto',
                   patience = 50, verbose=1, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, validation_split = 0.2, 
          batch_size = 10000, verbose = 1, epochs = 500)


#4 
results = model.evaluate(x_test, y_test)
print('loss', results[0])
print('acc', results[1])
print('mnist')

# loss 0.15299376845359802
# acc 0.9570000171661377

# Dense Layer
# loss 0.26949891448020935
# acc 0.9765999913215637

# LSTM
# loss 0.23475466668605804
# acc 0.9746999740600586

# Conv1D
# accuracy_score : 0.8602665435873766
# loss 0.3776072859764099
# f1 score 0.825052005839112
# run time 182.13
