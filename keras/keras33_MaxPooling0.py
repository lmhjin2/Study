import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)
# print(x_train)

x_train = x_train.reshape(60000, 28, 28, 1)
# print(x_train.shape[0]) # 60000
# x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_test.shape[0]) # 10000

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1)
# print(x_train.shape, x_test.shape)
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)


#2
model = Sequential()
model.add(Conv2D(8, kernel_size=(2,2), strides=2, # strides=보폭 / strides로 딱 못나누면 나머지는 버림. 
                 input_shape = (28, 28, 1), padding='same',activation='relu')) # (n, 27, 27, 9)
model.add(MaxPooling2D())
model.add(Conv2D(filters= 7, kernel_size=(2,2))) # (n, 25, 25, 10) # (3*3*9+1)*10  
model.add(Conv2D(15, (2,2))) # (n, 22, 22, 15)  # (a,a)일때 a-1개 만큼 (n, b, b, c) b에서 빠짐.
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu')) 
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
# (n, 27, 27, 10)
    # Conv2D(32, (3, 3), input_shape=(28, 28, 1))와 같은 레이어가 있다면:
    # 파라미터 수=(3×3×1+1)×32=320
    # (4*4*10+1)*15
model.summary()
# 3 x 3(필터 크기) x 3 (입력 채널(RGB)) x 32(출력 채널) + 32(출력 채널 bias) = 896
# shape = (batch_size, rows, columns, channels)
# shape = (batch_size, heights, widths, )
#3
es = EarlyStopping(monitor = 'val_loss', mode = 'auto',
                   patience = 50, verbose=1, restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, validation_split = 0.2, 
          batch_size = 3000, verbose = 1, epochs = 500)


#4 
results = model.evaluate(x_test, y_test)
print('loss', results[0])
print('acc', results[1])

# loss 0.2325105518102646
# acc 0.9404000043869019

# loss 0.18292994797229767
# acc 0.9462000131607056

# loss 0.15299376845359802
# acc 0.9570000171661377