import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,mean_absolute_error
from sklearn.datasets import fetch_california_housing

plt.rcParams['font.family'],"Malgun Gothic"
plt.rcParams['axes.unicode_minus']=False

#1 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state=0 )

#2
model = Sequential()
model.add(Dense(16, input_dim = 8))
model.add(Dense(32))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(8))
model.add(Dense(1))

#3
model.compile(loss='mse',optimizer='adam',
              metrics = ['mse','msle', 'mae'])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience = 300,verbose = 1,
                   restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=3000,
                 batch_size = 301, validation_split=0.2,
                 verbose = 2, callbacks=[es])

#4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test,y_predict))

rmse = RMSE(y_test, y_predict)
# rmsle = RMSLE(y_test, y_predict)
# print("음수갯수 :", y[y['count']<0].count())  ## 진짜 중요함 ##

print('loss:', loss)
print('RMSE:', rmse)
# print('RMSLE:', rmsle)
print('R2:',r2)
'''
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color = 'red',
         label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], color = 'blue',
         label = 'val_loss', marker = '.')
plt.legend(loc='upper right')
plt.title('캘리포니아 로스')
plt.xlabel('epochs')
plt.ylabel('val_loss')
plt.grid()
plt.show()
'''

# loss: [0.5937129259109497, 0.5937129259109497, 0.05482317879796028, 0.5589895248413086]
# RMSE: 0.7705276989046251
# R2: 0.5519150654481986

