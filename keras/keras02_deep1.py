from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성   예측값 소수점4자리까지
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10000))
model.add(Dense(1))


# 로스 :  2.1458981791511178e-05
# 1/1 [==============================] - 0s 49ms/step
# 예측값 :  [[3.99995]]


#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')
model.fit(x, y, epochs = 100)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 : ", loss)
result = model.predict([4])
print("예측값 : ", result)


# 로스 :  0.5749719142913818
# 1/1 [==============================] - 0s 67ms/step
# 에측값:  [[2.5301907]]

# 로스 :  0.010140781290829182
# 1/1 [==============================] - 0s 51ms/step
# 에측값:  [[3.832637]]

# 로스 :  0.13068611919879913
# 1/1 [==============================] - 0s 52ms/step
# 에측값 :  [[4.9247518]]

# test