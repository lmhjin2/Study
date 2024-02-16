# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import time as tm

#1 
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path+"train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
submission_csv = pd.read_csv(path + "submission.csv")

# print(x.shape, y.shape) # (1459, 9), (1459,)
# print(train_csv.isna().sum()) 결측치 확인
train_csv = train_csv.dropna()  # 결측치가 존재하는 행 전체 삭제.
# print(train_csv.shape)  # (1328, 10)
# print(test_csv.isna().sum())
test_csv = test_csv.fillna(test_csv.mean()) # 결측치는 평균처리
x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state = 65456 )
# print(x_train.shape, y_train.shape) # (1240, 9) (1240,)

#2 
model = Sequential()
model.add(Dense(18, input_dim = 9))
model.add(Dense(36))
model.add(Dense(53))
model.add(Dense(23))
model.add(Dense(12))
model.add(Dense(1))

#3
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train,
          epochs = 100, batch_size = 130,
          validation_split = 0.2, verbose = 2)

#4
loss = model.evaluate(x_test, y_test)
y_submit = model.evaluate(test_csv)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

submission_csv.to_csv(path + "submisson_val.csv", index = False)

print('loss:', loss)
print('r2:', r2)

# loss: 3801.428466796875
# r2: 0.456643443045472

