from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import time

# 현재 사이킷런버전 1.3.0 보스턴 안됨. 그래서 삭제할거임
# pip uninstall scikit-learn
# pip uninstall scikit-image
# pip uninstall scikit-learn-intelex
# pip install scikit-learn==1.1.3

datasets = load_boston()
# print(datasets)
x = datasets.data
y = datasets.target
# print(x)
# print(x.shape)  # (506, 13) 506행 13열

# print(y)
# print(y.shape)  # (506,)

print(datasets.feature_names)
# ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
print(datasets.DESCR)
# ['범죄율', ]

# 실습
# train_size 0.7 이상 0.9 이하
# R2 0.62 이상 / 0.8 이상

random_state_value = 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.3, random_state = random_state_value )

model = Sequential()
model.add(Dense(26, input_dim = 13))
model.add(Dense(52))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(13))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
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

import warnings
warnings.filterwarnings('ignore')

# random_state = 1, epochs = 8000, batch_size = 30
# 로스 : 18.694143295288086
# R2 스코어 : 0.796036779999476
# 걸린시간 :  58.3 초
# random_state = 1, epochs = 1000, batch_size = 30
# 로스 : 19.259849548339844
# R2 스코어 : 0.7898646249851895
# 걸린시간 :  7.9 초

