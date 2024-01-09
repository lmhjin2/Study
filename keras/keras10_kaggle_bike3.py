# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import time as tm

#1. 데이터
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = train_csv['count']

random_state_value = 1
train_size_value = 0.6

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size = train_size_value, random_state = random_state_value
)

x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, test_size = 0.5, random_state = 0)

#2. 모델

model = Sequential()
model.add(Dense(16, input_dim = 8, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(24, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))


#3. 컴파일 훈련
model.compile(loss = 'msle', optimizer = 'adam')
start_time = tm.time()
model.fit(x_train, y_train, epochs = 1000, batch_size = 381, verbose = 2, validation_data = (x_val, y_val))
end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)    # 그냥 전부 절대값 씌운거
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

submission_csv['count'] = y_submit

submission_csv.to_csv(path + "submission_0109_val.csv", index = False)

print("음수갯수 :", submission_csv[submission_csv['count']<0].count())
print("R2 : ", r2)
print("run time :", run_time)
print("random state value : ", random_state_value)
print("train size value : ", train_size_value)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
rmsle = RMSLE(y_test, y_predict)
print("MSE : ", loss)
print("RMSE", rmse)
print("RSMLE", rmsle)
































