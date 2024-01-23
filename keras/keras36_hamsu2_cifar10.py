import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import time as tm

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)
# print(np.unique(y_train, return_counts=True))
    # (array([0,     1,     2,    3,    4,    5,    6,   7,     8,    9], dtype=uint8),
    #  array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
    #   dtype=int64))

# acc = 0.77 이상
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# import matplotlib.pyplot as plt
# plt.imshow(x_train[5], "gray")
# plt.show()

# x_train = x_train/255 # 0~255까지 있는 데이터라 255로 나눠서 mimaxscaler랑 같은효과
# x_test = x_test/255 

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

#2
# model = Sequential()

# model.add(Conv2D(50, kernel_size=(2,2), input_shape=(32,32,3),
#                  padding='same', strides=2,
#                  activation='sigmoid'))
# # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) # 사실상 기본값임. pool_size(2,2)가 기본. strides는 pool_size와 같은게 기본.
# model.add(Dropout(0.2))
# model.add(Conv2D(50, kernel_size=(2,2), padding='same', activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(50, kernel_size=(2,2), padding='same', activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(220))
# model.add(Dropout(0.2))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

#2 hamsu
input1 = Input(shape=(32,32,3))
conv1 = Conv2D(50, (2,2), padding='same', strides=2,
               activation='sigmoid')(input1)
max1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv1)
drop1 = Dropout(0.2)(max1)
conv2 = Conv2D(50, (2,2), padding='same', activation='relu')(drop1)
max2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv2)
drop2 = Dropout(0.2)(max2)
conv3 = Conv2D(50, (2,2), padding='same', activation='relu')(drop2)
max3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv3)
drop3 = Dropout(0.2)(max3)
flat = Flatten()(drop3)
dense1 = Dense(220)(flat)
drop4 = Dropout(0.2)(dense1)
drop5 = Dropout(0.5)(drop4)
output1 = Dense(10, activation='softmax')(drop5)

model = Model(inputs = input1, outputs = output1)

#3
es = EarlyStopping(monitor='val_accuracy', mode = 'auto',
                   patience = 10, verbose = 1,
                   restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer = 'adam',
              metrics=['accuracy'])
start_time = tm.time()
model.fit(x_train, y_train, epochs = 100, batch_size = 1818, 
          verbose = 1 , validation_split = 0.18 , callbacks=[es])
end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)

y_train = np.argmax(y_train, axis = 1)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis = 1)

acc = accuracy_score(y_test, y_predict)

print('run time', run_time)
print('loss', results[0])
print('acc ', results[1], acc)



# run time 6824.01
# loss 0.6461122035980225
# acc  0.7796000242233276 0.7796


# run time 294.33
# loss 1.1299116611480713
# acc  0.6051999926567078 0.6052


# run time 346.52
# loss 1.320745825767517
# acc  0.5952000021934509 0.5952

# hamsu
# run time 294.8
# loss 1.1331857442855835
# acc  0.6046000123023987 0.6046



# scaling 1_1 MinMax
# loss 1.1877038478851318
# acc  0.5782999992370605 0.5783

# scaling 1_2 Standard
# loss 1.3619595766067505
# acc  0.5141000151634216 0.5141

# scaling 2_1 MinMax
# loss 1.4194271564483643
# acc  0.4918999969959259 0.4919

# scaling 2_2 Standard
# loss 1.258912444114685
# acc  0.548799991607666 0.5488








