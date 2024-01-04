import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([range(10)])   # range는 python 제공 기본함수 # numpy 제공 함수는 보통 np가 붙음(내가 지정한 이름)
print(x)        # [[0 1 2 3 4 5 6 7 8 9]]   # 처음부터 10개
print(x.shape)  # (1,10)

x = np.array([range(1,10)]) # 1 부터 (10-1) 까지/ range(a,b) = a 부터 b-1 까지
print(x)        # [[1 2 3 4 5 6 7 8 9]]
print(x.shape)  # (1, 9)
# 이 위는 range 설명을 위한것

x = np.array([range(10), range(21,31), range(201, 211)]) # (0~9), (21~30), (201~210)
print(x)   
print(x.shape)  # (3, 10)
x = x.transpose()   # 행과 열을 바꿔주는 명령어 == x = x.T
print(x)
print(x.shape)  # (10, 3)

# 파이썬에서 [] 안에 들어가있는걸 list 라고 함. 두개 이상은 list 데이터 형태
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])    # (3, 10)
print(y.shape)  # (3, 10)   
y = y.transpose()   # (10, 3)

model = Sequential()
model.add(Dense(3, input_dim = 3))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 3000, batch_size = 1)

loss = model.evaluate(x, y)
results = model.predict([[10, 31, 211]])    # (1,3)
print("로스 :", loss)
print("예측값 :", results)



# Dense 한줄, epochs = 3000, batch_size = 1
# 로스 : 2.7738476404692847e-08
# 예측값 : [[11.000001   1.9997574 -1.0000008]]

# Dense 한줄, epochs = 3000, batch_size = 1
# 로스 : 3.973624984610069e-09
# 예측값 : [[11.         1.9999053 -1.0000052]]


# 예측값 : [[11     2       -1]]

















