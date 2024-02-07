 # https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_error
import time as tm
from sklearn.svm import LinearSVR

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

#2
model = LinearSVR()

#3
start_time = tm.time()

model.fit(x_train, y_train)

end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4 
r2= model.score(x_test,y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
r21 = r2_score(y_test, y_predict)

submission_csv['count'] = y_submit
# submission_csv.to_csv(path + "submission_0116_scale_rbs.csv", index=False)

print('r21:', r21)
print('r2:', r2)
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

# r21: 0.18335540637638648
# r2: 0.18335540637638648
# run time: 0.0
