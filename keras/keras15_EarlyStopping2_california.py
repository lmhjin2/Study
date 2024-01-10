import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import time as tm
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

#1
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.3, random_state = 7 )

#2
model = Sequential()
model.add(Dense(16, input_dim = 8))
model.add(Dense(32))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(8))
model.add(Dense(1))

#3
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor = 'val_loss',
                   mode = 'auto',
                   patience = 15,
                   verbose = 1,
                   restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs = 300,
                 batch_size = 301, validation_split = 0.2,
                 verbose = 2,
                 callbacks=[es])

#4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('loss:', loss)
print('r2:', r2)

# print(hist.history['loss']) # 로스값이 들어있음.
# print(hist.history['val_loss'])
'''
plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], color = 'red',
        #  label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue',
         label = "val_loss", marker = '.')
plt.legend(loc = 'upper right')
plt.title('캘리포니아 로스값')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
'''

# loss: 0.7462538480758667
# r2: 0.44137737734148896

# restore_best_weight = True
# loss: 0.6875315308570862
# r2: 0.4853350152062813


