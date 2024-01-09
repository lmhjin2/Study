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

## 잘라라 1~10 train, 11~13 val, 14~16 test

x_train = x[:10]
y_train = y[:10]    # 1이 0번째, 10이 9번째 데이터

print(x_train, y_train)

x_val = x[10:13]
y_val = y[10:13]    # 11이 10번째 데이터

print(x_val, y_val)

x_test = x[13:16]
y_test = y[13:16]   # 14가 13번째 데이터

print(x_test, y_test)

#2. 모델구성

model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = tm.time()
model.fit(x_train, y_train, epochs = 300, batch_size = 1,
          validation_data = (x_val, y_val), verbose = 3)
end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
predict = model.predict([11000, 180])
r2 = r2_score(y_test, y_predict)
print("loss: ", loss)
print("r2 :", r2)
print("predict 11000, 180 : ", predict)







