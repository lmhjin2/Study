# 09_1 copy
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import time

datasets = load_boston()
# print(datasets)

x = datasets.data
y = datasets.target
# print(x.shape)  # (506, 13) 506행 13열
# print(y.shape)  # (506,)
# print(datasets.feature_names)
# ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
# print(datasets.DESCR)
# ['범죄율', '어쩌구',,,]

random_state_value = 1


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.3, random_state = random_state_value )

# 1. minmax 2. standard

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train))  # 0.0
# print(np.min(x_test))   # -0.03297872340425531
# print(np.max(x_train))  # 1.0
# print(np.max(x_test))   # 1.210017220702162

# 한번에 scaling 해도 각 column 별로 scaling 됨
# 나중엔 column 각각 지정해서 다른 scaling 함.

model = Sequential()
model.add(Dense(26, input_dim = 13))
model.add(Dense(52))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(13))
model.add(Dense(1))


model.compile(loss='mae', optimizer='adam')
start_time = round(time.time(), 2)
model.fit(x_train, y_train, epochs = 1000, batch_size=30)
end_time = round(time.time(), 2)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("로스 :", loss)
# print("예측값 :", y_predict)
print("R2 스코어 :", r2)
print("걸린시간 : ", round(end_time - start_time, 2), "초")
print("Random State:", random_state_value)
print("boston")

import warnings
warnings.filterwarnings('ignore')


# random_state = 1, epochs = 8000, batch_size = 30, loss = 'mse'
# 로스 : 18.694143295288086
# R2 스코어 : 0.796036779999476
# 걸린시간 :  58.3 초

# scaler = MinMaxScaler()
# 로스 : 3.1486480236053467
# R2 스코어 : 0.7835180345040087
# 걸린시간 :  9.9 초

# scaler = StandardScaler()
# 로스 : 3.144235610961914
# R2 스코어 : 0.7939326089286648
# 걸린시간 :  9.63 초

# scaler = MaxAbsScaler()
# 로스 : 3.134629487991333
# R2 스코어 : 0.7852473736299466
# 걸린시간 :  9.71 초

# scaler = RobustScaler()
# 로스 : 3.1448447704315186
# R2 스코어 : 0.7995639930028042
# 걸린시간 :  9.69 초


# RoubustScaler
