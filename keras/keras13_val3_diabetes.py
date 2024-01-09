import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.datasets import load_diabetes

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) (442,10) (442,)
x_train , x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state = 9 )

#2
model = Sequential()
model.add(Dense(20, input_dim = 10))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 40,
          validation_split = 0.2, verbose = 2)

#4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('loss: ', loss)
print('r2 : ', r2)

# loss:  2152.1103515625
# r2 :  0.5768195938083853


