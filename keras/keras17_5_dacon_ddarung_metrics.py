# https://dacon.io/competitions/open/235576/mysubmission
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,mean_absolute_error

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.dropna()  # 결측치 드랍.
test_csv = test_csv.fillna(test_csv.mean()) # 결측치에 평균치넣기
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, random_state=151717 )

#2
model = Sequential()
model.add(Dense(828, input_dim = 9))
model.add(Dense(560))
model.add(Dense(330))
model.add(Dense(180))
model.add(Dense(108))
model.add(Dense(52))
model.add(Dense(27))
model.add(Dense(1))

#3
model.compile(loss = 'mse', optimizer='adam',
              metrics=['mse','msle','mae'])
es = EarlyStopping(monitor = 'val_loss',
                   mode = 'auto', patience = 200,
                   restore_best_weights = True )
hist = model.fit(x_train, y_train, epochs = 3000,
                 batch_size = 25, validation_split = 0.13,
                 verbose=3, callbacks=[es])

#4
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

submission_csv['count']=y_submit
submission_csv.to_csv(path+"submission_0110_met.csv",index=False)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
# rmsle =RMSLE(y_test, y_predict)

print('loss:', loss)
print('RMSE:', rmse)
# print('RMSLE:', rmsle)
print('r2:', r2)
print('따릉')
'''
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color = 'red',
         label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue',
         label = 'val_loss', marker='.')
plt.legend(loc = 'upper right')
plt.title("따릉이 로스")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
'''
# loss: [2494.04052734375, 2494.04052734375, 0.6432426571846008, 39.16857147216797]
# RMSE: 49.94036818182851
# r2: 0.6079631835073899








