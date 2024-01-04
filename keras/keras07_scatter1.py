import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10]) 
y = np.array([1,2,3,4,6,5,7,8,9,10])
# x = np.array([range(1,11)])

# [검색] train과 test를 섞어서 7:3으로 자를 수 있는 방법을 찾아라
# 힌트 : 사이킷런

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 3)

'''
print(x_train)
print(x_test)
print(y_train)
print(y_test)
'''

model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 3000, batch_size = 7)

loss = model.evaluate(x_test, y_test)
results = model.predict(x)

print("로스 :", loss)
print("예측값 :", results)



# 로스 : 0.6666666865348816
# 예측값 : [[11.]]


import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.plot(x, results, color = 'red')
plt.show()





