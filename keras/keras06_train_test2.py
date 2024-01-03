import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10]) 
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [실습] 넘파이 리스트의 슬라이싱!! 7:3 으로 잘라라
x_train = x[0:7:1]  # [1,2,3,4,5,6,7]
y_train = y[0:7:1]  # [1,2,3,4,5,6,7]
# 60% ~ 90% 는 훈련

x_test = x[7:10:1]  # [8, 9, 10]
y_test = y[7:10:1]  # [8, 9, 10]
# 나머지로 결과값 테스트

print(x_train)
print(y_train)
print(x_test)
print(y_test)



#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 3000, batch_size = 1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([11000, 7])
print("로스 :", loss)
print("예측값 :", results)


