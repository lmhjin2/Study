import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# #1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,6,5,7])
# 60% ~ 90% 는 훈련

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])
# 나머지로 결과값 테스트

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 3000, batch_size = 1)

loss = model.evaluate(x_test, y_test)
results = model.predict([11000, 7])
print("로스 :", loss)
print("예측값 :", results)
