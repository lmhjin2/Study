# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error
import matplotlib.pyplot as plt

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

x = train_csv.drop(['casual', 'registered', 'count'],
                   axis = 1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 1234 )

#2
model = Sequential()
model.add(Dense(160, input_dim = 8, activation = 'relu'))    # relu 는 0이하는 0으로, 양수는 그대로 뽑아내는것
model.add(Dense(128, activation = 'relu'))
model.add(Dense(92, activation = 'relu'))
model.add(Dense(48, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

#3 
model.compile(loss='msle', optimizer='adam') 

from keras.callbacks import EarlyStopping
# val_loss : 1.3645
# R2:  0.23980802276182256
# RMSLE: 1.1377991895665427
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience = 28,
                   verbose=1)

hist = model.fit(x_train, y_train, epochs = 1000,
                 batch_size = 154, validation_split=0.18,
                 verbose = 2, callbacks=[es])

#4
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)    # 원래 하던거. 근데 count에 음수가 나옴;;
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_0109_es.csv", index = False)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
rmsle = RMSLE(y_test, y_predict)
# print("음수갯수 :", submission_csv[submission_csv['count']<0].count())  ## 진짜 중요함 ##
print("R2: ", r2)
# print("MSE:", loss)
# print("RMSE:", rmse)
print("RMSLE:", rmsle)
'''
plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], color = 'red',
#          label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue',
         label = 'val_loss', marker = '.')
plt.legend(loc='upper right')
plt.title("바이크 로스")
plt.xlabel('epoch')
plt.ylabel('val_loss')
plt.grid()
plt.show()
'''

