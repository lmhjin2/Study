# 원핫하고, 모델 완성해서 돌려라!!!
# acc = 0.98 이상

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score
import time as tm

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)
# print(x_train)

# x_train = x_train.reshape(60000, 28, 28, 1)
# print(x_train.shape[0]) # 60000
# x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_test.shape[0]) # 10000

# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1)
# print(x_train.shape, x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.dtype, y_test.dtype)


# scaling 1_1 MinMax
# x_train = x_train/255
# x_test = x_test/255

# scaling 1_2 Standard
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5

# scaling 2_1 MinMax
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)
# x_train = x_train.reshape(60000, 28,28,1)
# x_test = x_test.reshape(10000, 28,28,1)

# scaling 2_2 Standard
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000, 28,28,1)


#2

# model = Sequential()
# model.add(Conv2D(9, kernel_size=(2,2), input_shape = (28, 28, 1), 
#                  padding='same', strides=2,
#                  activation='sigmoid')) # (n, 27, 27, 9)
# model.add(MaxPooling2D())
# model.add(Conv2D(10, kernel_size=(2,2), activation='relu')) # (n, 25, 25, 10) # (3*3*9+1)*10  
# model.add(Conv2D(15, (2,2), padding='same', activation='relu')) # (n, 22, 22, 15)  # (a,a)일때 a-1개 만큼 (n, b, b, c) b에서 빠짐.
# model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu')) 
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='softmax'))

#2 hamsu
input1 = Input(shape=(28,28,1))
conv2d1 = Conv2D(9, (2,2), padding='same', strides=2, 
                activation='sigmoid')(input1)
maxpool1 = MaxPooling2D()(conv2d1)
conv2d2 = Conv2D(10, (2,2), activation='relu')(maxpool1)
conv2d3 = Conv2D(15, (2,2), padding='same', activation='relu')(conv2d2)
flatten = Flatten()(conv2d3)
dense1 = Dense(10, activation='relu')(flatten)
dense2 = Dense(10, activation='relu')(dense1)
output1 = Dense(10, activation='softmax')(dense2)

model = Model(inputs = input1, outputs = output1)



#3
es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1,
                   patience = 10 , restore_best_weights=True )
# mcp = ModelCheckpoint(monitor='val_accuracy', mode = 'auto', verbose=1)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = tm.time()
model.fit(x_train, y_train, validation_split = 0.15, 
          batch_size = 300, verbose = 1, epochs = 100 , callbacks=[es])
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

# loss: 0.0266 - accuracy: 0.9930

#  run time 135.32
# loss 0.16735179722309113
# acc 0.979200005531311 0.9792

# run time 34.02
# loss 0.0694819912314415
# acc 0.9853000044822693 0.9853

# hamsu
# run time 69.34
# loss 0.1229056864976883
# acc 0.9646999835968018 0.9647




# 1_1
# loss 0.16487731039524078
# acc 0.9480999708175659 0.9481

# 1_2
# run time 36.97
# loss 0.08779416233301163
# acc 0.9721999764442444 0.9722

# 2_1
# run time 26.7
# loss 0.10805419832468033
# acc 0.9664999842643738 0.9665

# 2_2
# run time 34.97
# loss 0.09090783447027206
# acc 0.9722999930381775 0.9723
