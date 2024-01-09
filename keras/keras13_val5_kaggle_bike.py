import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import time as tm

#1. 
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.74, random_state = 0 )

#2.
model = Sequential()
model.add(Dense(16, input_dim = 8, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(24, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

#3
model.compile(loss = 'msle', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 381,
          validation_split = 0.5, verbose = 2)

#4
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_0109_val.csv", index = False)

print('음수갯수 :', submission_csv[submission_csv['count']<0].count())
print('r2:', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
rmsle = RMSLE(y_test, y_predict)

print('MSE: ', loss)
print('RMSE:', rmse)
print('RMSLE:', rmsle)

# r2: 0.1687794739299654
# MSE:  1.4089157581329346
# RMSE: 164.88264420822574
# RMSLE: 1.1869775892885497

