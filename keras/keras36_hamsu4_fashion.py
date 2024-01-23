import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import time as tm

# 0.99 이상 -> 0.95 이상
#1
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)
print(np.unique(y_train, return_counts=True))
# 0부터 9까지 각각 6000개씩

# import matplotlib.pyplot as plt
# plt.imshow(x_train[1], "gray")
# plt.show()

# import matplotlib.pyplot as plt
# plt.imshow(x_train[1], "RdGy_r")
# plt.show()
# x_train = x_train.reshape(-1,28,28,1)
# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
# # x_train = x_train.reshape(60000,28,28,1)
# # x_test = x_test.reshape(10000,28,28,1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
# print(x_train.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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
# model=Sequential()
# model.add(Conv2D(16, (2,2), input_shape=(28,28,1), 
#                  # padding='same', strides=2,
#                  activation='sigmoid'))
# model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
# # model.add(MaxPooling2D())
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,2), activation='relu'))
# # model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(82, activation='relu'))
# model.add(Dense(46, activation='relu'))
# model.add(Dense(10, activation='softmax'))

#2 hamsu
input1 = Input(shape=(28,28,1))
conv1 = Conv2D(16, (2,2), padding='same', strides=2,
               activation='sigmoid')(input1)
conv2 = Conv2D(32, (2,2), padding='same', activation='relu')(conv1)
max1 = MaxPooling2D()(conv2)
drop1 = Dropout(0.2)(max1)
conv3 = Conv2D(16, (2,2), activation='relu')(drop1)
max2 = MaxPooling2D()(conv3)
flatt = Flatten()(max2)
dense1 = Dense(82, activation='relu')(flatt)
dense2 = Dense(46, activation='relu')(dense1)
output1 = Dense(10, activation='softmax')(dense2)

model = Model(inputs = input1, outputs = output1)


#3
es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience = 10, verbose = 1, restore_best_weights = True)
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
start_time=tm.time()
model.fit(x_train, y_train, epochs = 100, batch_size = 500,
          verbose=1, validation_split = 0.21 , callbacks=[es])
end_time=tm.time()
run_time = round(end_time-start_time, 2)

#4
results = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

y_train = np.argmax(y_train, axis = 1)
y_test = np.argmax(y_test, axis = 1)
y_predict = np.argmax(y_predict, axis = 1)

acc = accuracy_score(y_test, y_predict)

print('run time ', run_time)
print('loss', results[0])
print('acc', results[1], acc)


# run time  159.16
# loss 0.23071454465389252
# acc 0.9204999804496765 0.9205

# run time  63.41
# loss 0.22961604595184326
# acc 0.9218000173568726 0.9218

# 위가 기본
# ==================================================================
# 아래가 padding + strides

# run time  46.13
# loss 0.3341246247291565
# acc 0.8773000240325928 0.8773

# run time  53.36
# loss 0.23424088954925537
# acc 0.916100025177002 0.9161

# ====================================================================
# 아래는 no MaxPooling

# run time  49.1
# loss 0.2532179653644562
# acc 0.9108999967575073 0.9109


# hamsu
# run time  42.47
# loss 0.2569473385810852
# acc 0.9071000218391418 0.9071




# scaling 1_1 MinMax
# loss 0.30217477679252625
# acc 0.8898000121116638 0.8898

# scaling 1_2 Standard
# loss 0.29266855120658875
# acc 0.8931999802589417 0.8932

# scaling 2_1 MinMax
# loss 0.3006904423236847
# acc 0.8896999955177307 0.8897

# scaling 2_2 Standard
# loss 0.29515236616134644
# acc 0.8913999795913696 0.8914


