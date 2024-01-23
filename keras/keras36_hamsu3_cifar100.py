import numpy as np
import pandas as pd
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import time as tm

# accuracy 0.4 이상

# 1
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3), (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3), (10000, 1)
# print(np.unique(y_train, return_counts=True))
# 0 부터 99 전부 500개

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# scaling 1_1 MinMax
# x_train = x_train/255
# x_test = x_test/255

# scaling 1_2 Standard
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5

# scaling 2_1 MinMax
# x_train = x_train.reshape(50000, 32*32*3)
# x_test = x_test.reshape(10000, 32*32*3)
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)
# x_train = x_train.reshape(50000, 32,32,3)
# x_test = x_test.reshape(10000, 32,32,3)

# scaling 2_2 Standard
x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_train = x_train.reshape(50000, 32,32,3)
x_test = x_test.reshape(10000, 32,32,3)



# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], "gray")
# plt.show()

# 2
# model = Sequential()
# model.add(Conv2D(24, (2,2), input_shape=(32,32,3), 
#                  padding='same', strides=2, activation='sigmoid'))
# # (16, 16, 24) -> (32, 32, 24)
# # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Conv2D(45, (2,2), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Conv2D(31, (2,2), activation='relu'))
# model.add(Dropout(0.2))
# # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Flatten())
# model.add(Dense(120, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(82, activation='relu'))
# model.add(Dense(100, activation='softmax'))

# model.summary()

#2 hamsu
input1 = Input(shape=(32,32,3))
conv1 = Conv2D(24, (2,2), padding='same', strides=2,
               activation='sigmoid')(input1)
maxpl1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv1)
conv2 = Conv2D(45, (2,2), padding='same', activation='relu')(maxpl1)
drop1 = Dropout(0.2)(conv2)
maxpl2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(drop1)
conv3 = Conv2D(31, (2,2), activation='relu')(maxpl2)
drop2 = Dropout(0.2)(conv3)
maxpl3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(drop2)
flatt = Flatten()(maxpl3)
dense1 = Dense(120, activation='relu')(flatt)
drop3 = Dropout(0.2)(dense1)
dense2 = Dense(82, activation='relu')(drop3)
output1 = Dense(100, activation='softmax')(dense2)

model = Model(inputs = input1, outputs= output1)

#3
es = EarlyStopping(monitor = 'val_accuracy', mode='auto',
                   patience = 10, verbose = 1, restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])

start_time = tm.time()

model.fit(x_train, y_train, epochs = 100, batch_size = 500,
          verbose = 1, validation_split = 0.15, callbacks = [es])

end_time = tm.time()
run_time = round(end_time-start_time, 2)

#4
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_train = np.argmax(y_train, axis = 1)
y_test = np.argmax(y_test, axis = 1)
y_predict = np.argmax(y_predict, axis = 1)

acc = accuracy_score(y_test, y_predict)

print('run time', run_time)
print('loss', results[0])
print('acc', results[1], acc)

# run time 217.21
# loss 2.3018598556518555
# acc 0.4108000099658966 0.4108

# run time 429.32
# loss 2.28546142578125
# acc 0.4097999930381775 0.4098

# run time 47.33
# loss 2.8813202381134033
# acc 0.29910001158714294 0.2991

# hamsu
# run time 52.56
# loss 3.0714077949523926
# acc 0.25589999556541443 0.2559


# scaling 1_1 MinMax
# loss 3.7503604888916016
# acc 0.1340000033378601 0.134

# scaling 1_2 Standard
# loss 3.217642307281494
# acc 0.23690000176429749 0.2369

# scaling 2_1 MinMax
# loss 3.5248875617980957
# acc 0.17870000004768372 0.1787

# scaling 2_2 Standard
# loss 3.2227888107299805
# acc 0.23659999668598175 0.2366
