import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size= 0.3, random_state= 7 )

#2
model = Sequential()
model.add(Dense(20, input_dim = 10))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3
model.compile(loss = 'mse', optimizer='adam')

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_loss',
                   mode = 'auto',   # min이나 max나 auto가 잡아줌
                   patience = 10,
                   verbose = 1,
                   restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=3000,
          batch_size = 35, validation_split=0.2,
          verbose = 2, callbacks=[es])

#4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

# print(hist.history['val_loss'])
print('loss:', loss)
print('r2:', r2)
'''
plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], color = 'red',
#          label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue',
         label = 'val_loss', marker = ',')
plt.legend(loc = 'upper right')
plt.title('당뇨병 로스값')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()
'''
# 기본
# loss: 2779.651123046875
# r2: 0.512714340577765

# restore_best_weight = True
# loss: 2737.075439453125
# r2: 0.5201781073002912

