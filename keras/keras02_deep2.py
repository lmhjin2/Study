from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])     # 4랑 5랑 위치 일부러 바꾼거임

#2 요 데이터를 훈련해서 최소의 Loss 값을 만들어라
### [실습] 100 epochs에 01_2번과 같은 결과를 빼라 / loss 0.33이하
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))



# 로스 :  0.32382264733314514
# 1/1 [==============================] - 0s 87ms/step
# 7의 예측값 :  [[6.792662]]

# 로스 :  0.3238254189491272
# 1/1 [==============================] - 0s 83ms/step
# 7의 예측값 :  [[6.79145]]





model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

loss = model.evaluate(x, y)
print("로스 : ", loss)
result = model.predict([7])
print("7의 예측값 : ", result)

# 로스 :  0.32380956411361694
# 1/1 [==============================] - 0s 50ms/step
# 7의 예측값 :  [[6.8002677]] epochs 10만

# 로스 :  0.32380953431129456
# 1/1 [==============================] - 0s 51ms/step
# 7의 예측값 :  [[6.8000946]] epochs 10만
