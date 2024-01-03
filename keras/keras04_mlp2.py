# [실습]
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [9,8,7,6,5,4,3,2,1,0]
             ]
             )
y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x)
print(x.shape, y.shape)  # x.shape = (2, 10) , y.shape = (10,)
x = x.T     # 행무시 열우선 ★★★
print(x.shape)  # x.shape (10, 2)  / 두개의 열=두개의 컬럼=두개의 속성=두개의 특성=두개의 차원

model = Sequential()
model.add(Dense(1, input_dim = 3))  #(n,3)
# y= w1x1 + w2x2 + w3x3 + b
# 레이어가 단순한게 더 좋을 때도 있음 / keras01_1.py 에 있는거

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs = 10000, batch_size = 10)

loss = model.evaluate(x,y)
results = model.predict([[10, 1.3, 0]])  # (1,3)
print("로스 :", loss)
print("예측값 :", results)


# model.add(Dense(1, input_dim = 3)) 한줄
# epochs=3000, batch_size = 1
# 로스 : 2.2670577221051147e-11
# 예측값 : [[9.999998]]  0.000002
