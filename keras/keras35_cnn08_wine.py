import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1
datasets = load_wine()
x = datasets.data
y = datasets.target

x = x.reshape(-1,13,1,1)

# print(x.shape, y.shape) # (178, 13) (178,)
# print(pd.value_counts(y))
    # 1    71
    # 0    59
    # 2    48
# print(np.unique(y, return_counts=True))
    # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from keras.utils import to_categorical
y_encoded = to_categorical(y)
# print(y)    # [0 0 0 0 0.... 1 1 1 1.... 2 2 2 2....]
# print(y_encoded)    # [[1. 0. 0.]... [0. 1. 0.]....[0. 0. 1.]]
# print(y_encoded.shape)  # (178, 3)
# 마지막 Dense(Layer)에 3 

x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, stratify=y, train_size = 0.8, random_state = 4 )
# print(y_test)   # y_encoded 랑 독같이 생김
# print(y_test.shape) # (36,3) # y_encoded 랑 독같이 생김

# print(np.unique(y_test, return_counts=True))   
    # (array([0., 1.], dtype=float32), array([72, 36], dtype=int64))
# [0. 1. 0.] 이렇게 생겨서 0두개 1하나.   2:1 비율이 맞음

#2
model = Sequential()
model.add(Conv2D(33, (1,1),input_shape=(13,1,1),
                 padding='same',
                 activation='relu'))
model.add(Conv2D(15, (2,2), padding='same', activation='relu'))
model.add(Conv2D(17, (2,2), padding='same', activation='relu'))
model.add(Conv2D(8, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))



#3
model.compile(loss = 'categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience = 1000, verbose=1,
                   restore_best_weights=True)

import time as tm
start_time = tm.time()
hist = model.fit(x_train, y_train, epochs = 1000,
                 batch_size = 96, validation_split = 0.2,
                 verbose = 1)
end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
# print(results) 치면 숫자 두개나오는데 보기 좋으라고 밑에처럼 하는거임.
print('loss:', results[0])
print('acc:', results[1])
# print(y_predict)    # [0.1 0.2 0.7] 의 형태
# print(y_test)       # [0 1 0] 의 형태
# print(y_predict.shape, y_test.shape)    # 둘다 (36, 3)

# OneHot은 axis 무조건 1
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

# print(y_test)
#   # [1 0 1 0 1 2 2 0 1 0 2 2 0 2 1 0 2 2 1 0 1 2 2 1 2 0 1 0 1 1 0 0 1 1 0 1]
# print(y_test.shape) # (36,)
# print(y_predict)
#   # [1 0 1 1 1 0 2 0 1 0 2 1 0 2 0 0 2 2 1 0 1 0 2 1 1 0 1 0 1 1 0 0 1 1 0 1]
# print(y_predict.shape)  # (36,)

acc = accuracy_score(y_predict, y_test)
print('accuracy_score :', acc)
print("run time:", run_time)



# loss: 0.0532647967338562
# acc: 1.0
# accuracy_score : 1.0


# scaler = MinMaxScaler()
# loss: 0.027338437736034393
# acc: 1.0
# accuracy_score : 1.0


# scaler = StandardScaler()
# loss: 0.018835054710507393
# acc: 1.0
# accuracy_score : 1.0


# scaler = MaxAbsScaler()
# loss: 0.027734629809856415
# acc: 1.0
# accuracy_score : 1.0


# scaler = RobustScaler()
# loss: 0.026079488918185234
# acc: 0.9722222089767456
# accuracy_score : 0.9722222222222222




# StandardScaler

# CPU
# 19.34 초

# GPU
# 18.95 초

# CNN
# loss: 0.03656196966767311
# acc: 0.9722222089767456
# accuracy_score : 0.9722222222222222
# run time: 24.21