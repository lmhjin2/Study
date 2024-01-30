import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler,MaxAbsScaler,StandardScaler
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

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32*3, 32)
x_test = x_test.reshape(10000, 32*3, 32)
# print(x_train.shape, x_test.shape)
# (50000, 3072) (30000, 3072)



#2
model = Sequential()
model.add(LSTM(15, input_shape=(32*3,32), activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(26, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(23, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(28, activation='swish'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3
es = EarlyStopping(monitor='val_accuracy', mode = 'auto',
                   patience = 50, verbose = 1,
                   restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer = 'adam',
              metrics=['accuracy'])
start_time = tm.time()
model.fit(x_train, y_train, epochs = 1000, batch_size = 18181, 
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

# Dense Layer
# loss 1.4936628341674805
# acc  0.5188999772071838 0.5189

# LSTM
# run time 56.11
# loss 2.1593985557556152
# acc  0.20020000636577606 0.2002
