import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

#2. 모델 구성 후 그려보기

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=70)


print(x_train)
print(x_test)
print(y_train)
print(y_test)


model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test)
results = model.predict(x)
print("로스 : ", loss)
# print("예측값: " , results)

# 시각화
plt.scatter(x,y)
plt.plot(x, results, color = 'red')
plt.show()



# 14/14 [==============================] - 0s 1ms/step - loss: 16.8640
# 1/1 [==============================] - 0s 61ms/step - loss: 12.4742

# random_state = 70
# 14/14 [==============================] - 0s 1ms/step - loss: 9.2090
# 1/1 [==============================] - 0s 70ms/step - loss: 16.9311




