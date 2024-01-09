import warnings
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import time as tm

warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston


datasets = load_boston()
# print(load_boston)
x = datasets.data
y = datasets.target
# print(x.shape)  # (506, 13)
# print("=====================================================================")
# print(y.shape)  # (506,)
# print(datasets.feature_names)
# print(datasets.DESCR)
random_state_value = 1
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = random_state_value)

#2. 모델 

model = Sequential()
model.add(Dense(26, input_dim = 13))
model.add(Dense(52))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(13))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train,
          epochs = 1000, batch_size = 30,
          validation_split = 0.2, verbose = 2)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('loss:', loss)
print('r2 : ', r2)


# loss: 23.26984214782715
# r2 :  0.7645405560466554




