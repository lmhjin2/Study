import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

#2. 모델 구성 후 그려보기.  

x_train, x_test, y_train, y_test = train_test_split(x, y,
            test_size=0.3, shuffle=True, random_state=5)

'''
print(x_train)
print(x_test)
print(y_train)
print(y_test)
'''

model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(10000))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3000, batch_size=1)

loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 스코어 : " , r2)

'''
results = model.predict(x)

plt.scatter(x,y)
plt.plot(x, results, color = 'red')
plt.show()
'''



# random_state = 2, epochs = 3000, batch_size = 14
# 로스 :  5.333738327026367
# R2 스코어 :  0.8749905076693789

# random_state = 2, epochs = 3000, batch_size = 1
# 로스 :  3.6478021144866943
# R2 스코어 :  0.9145046393755593

# random_state = 91, epochs = 3000, batch_size = 14
# 로스 :  29.316415786743164
# R2 스코어 :  -4.049718086596429




