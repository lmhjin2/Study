import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 1000, batch_size = 1)

loss = model.evaluate(x, y)
results = model.predict([11000, 7])
print("로스 :", loss)
print("예측값 :", results)

# Dense 한줄, epochs = 3000, batch_size = 1
# 로스 : 0.0
# 예측값 : [[100000.]]

# Dense 한줄, epochs = 1000, batch_size = 1
# 로스 : 2.0605739337042905e-13
# 예측값 : [[10999.999]]
