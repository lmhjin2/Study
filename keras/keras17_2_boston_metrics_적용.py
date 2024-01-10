import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,mean_absolute_error
from sklearn.datasets import load_boston

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state= 1 )

model = Sequential()
model.add(Dense(26, input_dim = 13))
model.add(Dense(52))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(13))
model.add(Dense(1))

model.compile(loss= 'mse', optimizer='adam',
              metrics=['mse','msle','mae'])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience = 30 ,
                   verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000,
                 batch_size=154, validation_split=0.18,
                 verbose=2,callbacks=[es])

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
rmsle = RMSLE(y_test, y_predict)

print('loss:', loss)
print('RMSE:', rmse)
# print('RMSLE:',rmsle)
print('R2:', r2)


# loss: [48.147987365722656, 48.147987365722656, 0.2021975815296173, 5.089956283569336]
# RMSE: 6.938875301306562
# R2: 0.47827065470306884









