import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([range(10)]) # (0~9) # (1,10)
x = x.transpose()   # (10, 1)

# 파이썬에서 [] 안에 들어가있는걸 list 라고 함. 두개 이상은 list 데이터 형태
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])    # (3, 10)
y = y.transpose()   # (10, 3)
# 예측 [10]

model = Sequential()
model.add(Dense(3, input_dim=1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 3000, batch_size = 1)

loss = model.evaluate(x, y)
results = model.predict([[10]])
print("로스 :", loss)
print("예측값 :", results)

# Dense 한줄, epochs = 3000, batch_size = 1
# 로스 : 1.1007845676647143e-11
# 예측값 : [[11.         2.0000083 -0.9999914]]




