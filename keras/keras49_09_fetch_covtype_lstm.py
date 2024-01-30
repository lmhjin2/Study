import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time as tm


datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(np.unique(x, return_counts=True))
#     # (array([-173., -166., -164., ..., 7168., 7172., 7173.]),
#     #  array([1, 2, 1, ..., 1, 1, 1], dtype=int64))
# print('=====================================================================')
# print(np.unique(y, return_counts=True))
    # (array([1, 2, 3, 4, 5, 6, 7]),
    #  array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
    #  dtype=int64))

# print(x.shape, y.shape) # (581012, 54)  (581012,)
# print(pd.value_counts(y))
    #   원래 순서       # 내가 걍 맞춘거
    # 2    283301       # 1    211840       
    # 1    211840       # 2    283301
    # 3     35754       # 3     35754
    # 7     20510       # 4      2747
    # 6     17367       # 5      9493
    # 5      9493       # 6     17367
    # 4      2747       # 7     20510

from keras.utils import to_categorical
y_ohe = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y_ohe, stratify=y, train_size = 0.8, random_state = 0 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

# print(x_train)
# print(y_train)

x_train = x_train.reshape(-1,54,1)
x_test = x_test.reshape(-1,54,1)

#2
model = Sequential()
model.add(LSTM(12, input_shape = (54,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='softmax'))
# keras만 8개. column이 하나 늘어나버림. 근데 7갠데 8개짜리로 뽑으면 여러가지로 귀찮아짐. 성능도 안좋음(보편적)

#3
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   # 월일_시분

path1 = "c:/_data/_save/MCP/k28/09/"
filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
filepath = "".join([path1, 'k28_', date, '_', filename])

model.compile(loss = 'categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience = 100, verbose=1,
                   restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss',mode='auto',
            verbose = 1, save_best_only=True,
            filepath=filepath)
start_time = tm.time()
hist = model.fit(x_train, y_train, epochs = 500,
                 batch_size = 50000, validation_split = 0.2,
                 verbose = 2 )
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
print('run time', run_time)

# loss: 0.3379572629928589
# acc: 0.8663201332092285
# accuracy_score : 0.8663201466399318
 


# scaler = MinMaxScaler()
# loss: 0.260427325963974
# acc: 0.9010868668556213
# accuracy_score : 0.9010868910441211


# scaler = StandardScaler()
# loss: 0.20535320043563843
# acc: 0.9242876768112183
# accuracy_score : 0.9242876689930553


# scaler = MaxAbsScaler()
# loss: 0.2867389917373657
# acc: 0.8877481818199158
# accuracy_score : 0.8877481648494445


# scaler = RobustScaler()
# loss: 0.20221827924251556
# acc: 0.9258022308349609
# accuracy_score : 0.9258022598383863

# RobustScaler  # MCP
# loss: 0.19374237954616547
# acc: 0.9310688972473145
# accuracy_score : 0.9310689052778327

 
# RobustScaler

# CPU
# 293.8 초

# GPU
# 68.57 초

# loss: 0.7334080338478088
# acc: 0.6861870884895325
# accuracy_score : 0.6861871036031773
# run time 328.24
