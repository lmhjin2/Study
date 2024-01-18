# https://dacon.io/competitions/open/235576/mysubmission
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error, mean_absolute_error

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

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


#2
model = Sequential()
model.add(Dense(108, input_dim = 9))
model.add(Dropout(0.2))
model.add(Dense(90))
model.add(Dropout(0.2))
model.add(Dense(87))
model.add(Dropout(0.2))
model.add(Dense(73))
model.add(Dropout(0.2))
model.add(Dense(68))
model.add(Dropout(0.2))
model.add(Dense(57))
model.add(Dropout(0.2))
model.add(Dense(49))
model.add(Dropout(0.2))
model.add(Dense(36))
model.add(Dropout(0.2))
model.add(Dense(21))
model.add(Dropout(0.2))
model.add(Dense(11))
model.add(Dense(1))

#3
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   # 월일_시분

path1 = "c:/_data/_save/MCP/k28/04/"
filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
filepath = "".join([path1, 'k28_', date, '_', filename])

model.compile(loss = 'mse', optimizer='adam',
              metrics=['mse','msle','mae'])
es = EarlyStopping(monitor = 'val_loss',
                   mode = 'auto', patience = 200,
                   restore_best_weights = True )
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1, save_best_only=True,
    filepath=filepath)
hist = model.fit(x_train, y_train, epochs = 3000,
                 batch_size = 25, validation_split = 0.13,
                 verbose=3, callbacks=[es, mcp])

#4
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

submission_csv['count']=y_submit
submission_csv.to_csv(path+"submission_0117_scale.csv",index=False)

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


# scaler = MinMaxScaler()
# loss: [2403.452392578125, 2403.452392578125, 0.5717068910598755, 37.98446273803711]
# RMSE: 49.02501764490937
# r2: 0.6222026637380059

# scaler = StandardScaler()
# loss: [2379.362548828125, 2379.362548828125, 0.5877285003662109, 36.955265045166016]
# RMSE: 48.77871023779605
# r2: 0.6259893232430387

# scaler = MaxAbsScaler()
# loss: [2399.464111328125, 2399.464111328125, 0.5455405712127686, 37.7288703918457]
# RMSE: 48.984325076426565
# r2: 0.6228295748300605

# scaler = RobustScaler()
# loss: [2424.36328125, 2424.36328125, 0.6572142839431763, 38.33124923706055]
# RMSE: 49.23782410543493
# r2: 0.6189156804017729



# StandardScaler

