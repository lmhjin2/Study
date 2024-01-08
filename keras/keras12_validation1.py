# keras06_train_test1.py 카피
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# #1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,6])
# 60% ~ 90% 는 훈련

x_val = np.array([6,7])
y_val = np.array([5,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])
# 나머지로 결과값 테스트

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_data = (x_val, y_val), verbose = 3)
# verbose = 0 : 침묵
# verbose = 1 : 원래 보던 그 epoch. default값
# verbose = 2 : 프로그레스바 삭제
# verbose = 3 : epoch 만 나옴
# verbose 0,1,2 제외 나머지 : epoch 만 나옴
loss = model.evaluate(x_test, y_test)   # 첫번째 1/1 evaluate
results = model.predict([11000, 7])     # 두번째 1/1 fit
print("로스 :", loss)
print("예측값 :", results)

# Dense 1줄, epochs = 3000, batch_size = 1
# 로스 : 0.03919036313891411
# 예측값 : [[1.0569212e+04], [6.8829894e+00]]