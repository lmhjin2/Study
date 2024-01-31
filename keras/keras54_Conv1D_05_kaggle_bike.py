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
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape, x_test.shape)
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
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   # 월일_시분

path1 = "c:/_data/_save/MCP/k28/05/"
filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
filepath = "".join([path1, 'k28_', date, '_', filename])

model.compile(loss='mse', optimizer='adam',
              metrics=['mae','mse','msle'])
# es = EarlyStopping(monitor='val_loss',
#                    mode='auto',
#                    patience = 200 ,
#                    verbose = 1 , restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',
#                       save_best_only=True, verbose=1,
#     filepath=filepath)

start_time = tm.time()

hist = model.fit(x_train, y_train, epochs = 1000,
                 batch_size = 760 , validation_split = 0.18,
                 verbose = 3)

end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4 
loss = model.evaluate(x_test,y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_0116_scale_rbs.csv", index=False)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
rmsle = RMSLE(y_test, y_predict)
# print("음수갯수 :", submission_csv[submission_csv['count']<0].count())  ## 진짜 중요함 ##
print('r2:', r2)
# print('RMSE', rmse)
# print('RMSLE', rmsle)
print('loss :', loss)
print("run time:", run_time)
print('kaggle bike')


# scaler = MinMaxScaler()
# run time: 50.65
# r2: 0.37518150868450384
# loss : [20384.2578125, 104.17037200927734, 20384.2578125, 1.7026294469833374]

# MinMaxScaler

# CPU
# 52.15 초

# GPU
# 35.97 초

# LSTM
# r2: 0.30905386552735703
# loss : [22541.623046875, 104.77494812011719, 22541.623046875, 1.6236993074417114]
# run time: 103.95

# Conv1D
# r2: 0.35022335701438456
# loss : [21198.5, 102.29973602294922, 21198.5, 1.6154061555862427]
# run time: 39.6

