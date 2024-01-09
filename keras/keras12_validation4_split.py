import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_error
import time as tm

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, shuffle = False)   # random_state는 shuffle이 False일땐 어차피 안먹지만 써도 에러는 안뜸. 대신 무거워 지겠죠?
print(x_train, x_test)  # (1~13), (14,15,16)
print(y_train, y_test)  # (1~13), (14,15,16)

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim = 1))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_split = 0.3) # 통상적으로 evaluate loss > val loss > loss 순으로 신뢰함

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
predict = model.predict([11000, 7])
r2 = r2_score(y_test, y_predict)    # x_test의 답안지가 y_test 고 모델이 푼 문제 답이 y_predict임

print("loss :", loss)
print("r2 :", r2)
print("predict 11000, 7 :", predict)













