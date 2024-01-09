import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import time as tm

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

## train_test_split 으로 잘라라. train 10 , test 3, validation 3

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.625, shuffle = False)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size = 0.5, shuffle = False)

print(x_train, x_val, x_test)   # (1~10) (11,12,13) (14,15,16)
print(y_train, y_val, y_test)   # (1~10) (11,12,13) (14,15,16)

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim = 1))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = tm.time()
model.fit(x_train, y_train, epochs = 300, batch_size = 1, validation_data = (x_val, y_val), verbose = 3,)
end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
predict = model.predict([11000, 7])
r2 = r2_score(y_test, y_predict)    # x_test의 답안지가 y_test 고 모델이 푼 문제 답이 y_predict임

print("run time: ", run_time)
print("loss :", loss)
print("r2 :", r2)
print("predict 11000, 7 :", predict)

# 0.12345678901234567   소수점 아래 17번째 4하나. 0.3 + 4e-17
# 0.30000000000000004
# 부동소수점 연산 2진수로 소수점아래를 표현하기 빡세서 00000000000004가 나오는거임