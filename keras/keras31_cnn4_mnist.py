# 원핫하고, 모델 완성해서 돌려라!!!
# acc = 0.98 이상

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import time as tm

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)
# print(x_train)

x_train = x_train.reshape(60000, 28, 28, 1)
# print(x_train.shape[0]) # 60000
x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_test.shape[0]) # 10000

# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1)
# print(x_train.shape, x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.dtype, y_test.dtype)


#2
model = Sequential()
model.add(Conv2D(9, kernel_size=(2,2), input_shape = (28, 28, 1), activation='relu')) # (n, 27, 27, 9)
model.add(Conv2D(10, kernel_size=(3,3),input_shape=(27,27,9), activation='relu')) # (n, 25, 25, 10) # (3*3*9+1)*10  
model.add(Conv2D(15, (4,4), input_shape=(25,25,10), activation='relu')) # (n, 22, 22, 15)  # (a,a)일때 a-1개 만큼 (n, b, b, c) b에서 빠짐.
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu')) 
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
# (n, 27, 27, 10)
    # Conv2D(32, (3, 3), input_shape=(28, 28, 1))와 같은 레이어가 있다면:
    # 파라미터 수=(3×3×1+1)×32=320
    # (4*4*10+1)*15
# model.summary()
# 3 x 3(필터 크기) x 3 (입력 채널(RGB)) x 32(출력 채널) + 32(출력 채널 bias) = 896

#3
es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1,
                   patience = 100 , restore_best_weights=True )
# mcp = ModelCheckpoint(monitor='val_accuracy', mode = 'auto', verbose=1)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = tm.time()
model.fit(x_train, y_train, validation_split = 0.17, 
          batch_size = 300, verbose = 1, epochs = 1000 , callbacks=[es])
end_time=tm.time()
run_time=round(end_time - start_time, 2)

#4 
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)

y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test, y_predict)

print('run time', run_time)
print('loss', results[0])
print('acc', results[1], acc)


#  1s - loss: 0.0266 - accuracy: 0.9930

# loss 0.2538575232028961
# acc 0.9800999760627747 0.9801