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
from keras.optimizers import *
learning_rate = 1.0
model.compile(loss = 'mse', optimizer=Adam(learning_rate=learning_rate),
              metrics=['mse','msle','mae'])
hist = model.fit(x_train, y_train, epochs = 100,
                 batch_size = 25, validation_split = 0.13,
                 verbose=2)
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
# print('loss:', loss)
# print('r2:', r2)
# print('따릉')
print("lr : {0}, 로스 : {1}, R2 : {2}".format(learning_rate, loss, r2))

# scaler = StandardScaler()
# loss: [2379.362548828125, 2379.362548828125, 0.5877285003662109, 36.955265045166016]
# RMSE: 48.77871023779605
# r2: 0.6259893232430387

