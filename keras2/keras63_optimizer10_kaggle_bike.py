 # https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_error
import time as tm
#1
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path +"train.csv", index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, random_state = 1234 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# print(x_train.shape, x_test.shape)
x_train = x_train.reshape(-1,8,1)
x_test = x_test.reshape(-1,8,1)
test_csv = test_csv.reshape(-1,8,1)


#2
model = Sequential()
model.add(Conv1D(160, 5, input_shape = (8,1), activation = 'relu'))    # relu 는 0이하는 0으로, 양수는 그대로 뽑아내는것
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(92, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(48, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

#3
from keras.optimizers import *
learning_rate = 1.0
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate),
              metrics=['mae','mse','msle'])
hist = model.fit(x_train, y_train, epochs = 200,
                 batch_size = 760 , validation_split = 0.18,
                 verbose = 3)
#4 
loss = model.evaluate(x_test,y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_0312.csv", index=False)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
rmsle = RMSLE(y_test, y_predict)
# print("음수갯수 :", submission_csv[submission_csv['count']<0].count())  ## 진짜 중요함 ##
# print('r2:', r2)
# print('loss :', loss)
# print('kaggle bike')
print("lr : {0}, 로스 : {1}, R2 : {2}".format(learning_rate, loss, r2))

# scaler = MinMaxScaler()
# run time: 50.65
# r2: 0.37518150868450384
# loss : [20384.2578125, 104.17037200927734, 20384.2578125, 1.7026294469833374]

# Epochs 100
# lr : 1.0, 로스 : [70336.125, 194.19534301757812, 70336.125, 23.25289535522461], R2 : -1.1559434533665325
# lr : 0.1, 로스 : [27665.099609375, 118.58285522460938, 27665.099609375, 1.8729465007781982], R2 : 0.15200913315843434
# lr : 0.01, 로스 : [21632.5625, 105.08380889892578, 21632.5625, 1.6908128261566162], R2 : 0.33691829919003646
# lr : 0.001, 로스 : [22736.037109375, 109.34420013427734, 22736.037109375, 1.7700517177581787], R2 : 0.30309470912668646
# lr : 0.0001, 로스 : [25330.099609375, 118.91284942626953, 25330.099609375, 1.980500340461731], R2 : 0.22358177479102082

# Epochs 200
# lr : 1.0, 로스 : [70336.125, 194.19534301757812, 70336.125, 23.25289535522461], R2 : -1.1559434533665325
# lr : 0.1, 로스 : [70336.125, 194.19534301757812, 70336.125, 23.25289535522461], R2 : -1.1559434533665325
# lr : 0.01, 로스 : [22770.9140625, 105.1636962890625, 22770.9140625, 1.603546380996704], R2 : 0.30202576050983165
# lr : 0.001, 로스 : [21805.966796875, 107.76563262939453, 21805.966796875, 1.7639071941375732], R2 : 0.33160330522555437
# lr : 0.0001, 로스 : [23561.232421875, 113.29276275634766, 23561.232421875, 1.83171808719635], R2 : 0.2778008319382319