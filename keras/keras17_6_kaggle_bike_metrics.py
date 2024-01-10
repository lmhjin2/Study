# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
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

#2
model = Sequential()
model.add(Dense(160, input_dim = 8, activation = 'relu'))    # relu 는 0이하는 0으로, 양수는 그대로 뽑아내는것
model.add(Dense(128, activation = 'relu'))
model.add(Dense(92, activation = 'relu'))
model.add(Dense(48, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

#3
model.compile(loss='mse', optimizer='adam',
              metrics=['mae','mse','msle'])
es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience = 200 ,
                   verbose = 1 , restore_best_weights=True)
start_time = tm.time()
hist = model.fit(x_train, y_train, epochs = 5000,
                 batch_size = 760 , validation_split = 0.18,
                 verbose = 3 , callbacks=[es])
end_time = tm.time()
run_time = round(end_time - start_time, 2 )

#4 
loss = model.evaluate(x_test,y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_0110.csv", index=False)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
rmsle = RMSLE(y_test, y_predict)
# print("음수갯수 :", submission_csv[submission_csv['count']<0].count())  ## 진짜 중요함 ##
print('run time:', run_time)
print('r2:', r2)
print('RMSE', rmse)
print('RMSLE', rmsle)


# r2: 0.3547404211512003
# RMSE 145.09009092809526
# RMSLE 1.309696439264264














