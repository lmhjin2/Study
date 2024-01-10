import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
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

#2
model = Sequential()
model.add(Dense(20, input_dim = 10))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3
model.compile(loss = 'mse', optimizer='adam',
              metrics = ['mse','msle','mae'])
es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience=100, restore_best_weights=True, verbose = 1)
hist = model.fit(x_train, y_train, epochs = 3000, 
                 batch_size = 25, validation_split = 0.13,
                 verbose = 3, callbacks=[es])

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















































