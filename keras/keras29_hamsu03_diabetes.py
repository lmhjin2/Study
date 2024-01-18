import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_squared_log_error
from sklearn.datasets import load_diabetes

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state = 0 )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
# model = Sequential()
# model.add(Dense(20, input_dim = 10))
# model.add(Dense(40))
# model.add(Dropout(0.2))
# model.add(Dense(50))
# model.add(Dropout(0.2))
# model.add(Dense(30))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dense(1))


#2
input1 = Input(shape=(10,))
dense1 = Dense(20)(input1)
dense2 = Dense(40)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(50)(drop1)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(30)(drop2)
drop3 = Dropout(0.2)(dense4)
dense5 = Dense(10)(drop3)
output1 = Dense(1)(dense5)

model = Model(inputs = input1, outputs = output1)



#3
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   # 월일_시분

path = "c:/_data/_save/MCP/k28/03/"
filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
filepath = "".join([path, 'k28_', date, '_', filename])
model.compile(loss = 'mse', optimizer='adam',
              metrics = ['mse','msle','mae'])
es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience=100, restore_best_weights=True,
                   verbose = 1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1, save_best_only=True,
    filepath=filepath)

hist = model.fit(x_train, y_train, epochs = 3000, 
                 batch_size = 25, validation_split = 0.13,
                 verbose = 3, callbacks=[es, mcp])

#4
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
rmsle = RMSLE(y_test, y_predict)
    

print('loss:' , loss)
print('RMSE:', rmse)
print('RMSLE:', rmsle)

# loss: [3722.598388671875, 3722.598388671875, 0.17765571177005768, 47.66206741333008]
# RMSE: 61.01310099210362
# RMSLE: 0.4214922509776867


# scaler = MinMaxScaler()
# loss: [3740.314453125, 3740.314453125, 0.18052496016025543, 48.00529861450195]
# RMSE: 61.15810913472119
# RMSLE: 0.42488229680108314

# scaler = StandardScaler()
# loss: [3690.262451171875, 3690.262451171875, 0.17990046739578247, 47.92462921142578]
# RMSE: 60.74752928544592
# RMSLE: 0.4241467632371064

# scaler = MaxAbsScaler()
# loss: [3726.89697265625, 3726.89697265625, 0.1793489009141922, 47.58124542236328]
# RMSE: 61.048316136665555
# RMSLE: 0.42349604018902787

# scaler = RobustScaler()
# loss: [3777.06494140625, 3777.06494140625, 0.18301081657409668, 47.85576629638672]
# RMSE: 61.45782963449922
# RMSLE: 0.42779765259117974


# StandardScaler