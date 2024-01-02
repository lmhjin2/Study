import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]
             )
y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x.shape)  # (2,10)    # 행 : 데이터 갯수  # 열 : 종류     # 스칼라 10개 짜리 두묶음
print(y.shape)  # (10,)     
# x = x.T   행과 열을 바꿔주는 명령어. 아랫줄과 효과가 똑같음.
x = x.transpose()
# [[1,1], [2, 1.1], [3, 1.2], ... [10, 1.3]]
print(x.shape)  # (10, 2)

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

#3. 열, 컬럼, 속성, 특성, 차원 = 2 // 같다
# (행무시, 열우선) <= 외워

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs = 1000, batch_size = 1)


# 실습 : 소수점 둘째 자리 까지 맞춘다.
loss = model.evaluate(x,y)
results = model.predict([[10, 1.3]])
print("로스 : ", loss)
print("[10, 1.3]의 예측값 :", results)


# 로스 :  0.00017557063256390393
# [10, 1.3]의 예측값 : [[10.00519]]