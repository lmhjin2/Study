# 04_mpl1 copy
# MLP = 멀티 레이어 퍼셉트론
# Multy-Layer Perceptron

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]
             )
y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x.shape)  # (2,10)    # 행 : 데이터 갯수  # 열 : 종류     # 스칼라 10개 짜리 두묶음
print(y.shape)  # (10,)     
# x = x.T   행과 열을 바꿔주는 명령어. 아랫줄과 효과가 똑같음.
x = x.transpose()
# 이걸 쓰기 싫었다면 아래처럼 x 값 입력
# x = np.array ([[1,1], [2, 1.1], [3, 1.2], ... [10, 1.3]])
print(x.shape)  # (10, 2)  / 행 무시 열 우선이기 때문에 input_dim과 같은수가 열로 되게끔 세팅


#2. 모델 구성 (순차적)
# model = Sequential()
# model.add(Dense(10, input_shape=(2,)))
# model.add(Dense(9))
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(1))

#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 10)                30

#  dense_1 (Dense)             (None, 9)                 99

#  dense_2 (Dense)             (None, 8)                 80

#  dense_3 (Dense)             (None, 7)                 63

#  dense_4 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 280 (1.09 KB)
# Trainable params: 280 (1.09 KB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

#2. 모델 구성 (함수형)
input1 = Input(shape=(2,))
dense1 = Dense(10)(input1)
dense2 = Dense(9)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation='relu')(drop1)
dense4 = Dense(7)(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs = input1, outputs = output1)

model.summary()

#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 2)]               0

#  dense (Dense)               (None, 10)                30

#  dense_1 (Dense)             (None, 9)                 99

#  dense_2 (Dense)             (None, 8)                 80

#  dense_3 (Dense)             (None, 7)                 63

#  dense_4 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 280 (1.09 KB)
# Trainable params: 280 (1.09 KB)
# Non-trainable params: 0 (0.00 Byte)


'''
#3. 열, 컬럼, 속성, 특성, 차원 = 2 // 같다
# (행무시, 열우선) <= 외워

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs = 10, batch_size = 2)


# 실습 : 소수점 둘째 자리 까지 맞춘다.
loss = model.evaluate(x,y)
results = model.predict([[10, 1.3]])    # [10,1.3] 은 (2,) / [[10, 1.3]] 은 (1,2) shape를 맞추기 위한 []
print("로스 : ", loss)
print("[10, 1.3]의 예측값 :", results)


# 로스 :  0.00017557063256390393
# [10, 1.3]의 예측값 : [[10.00519]]


'''