# 실습. R2 0.55 ~ 0.6 이상

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) # (20640,8) / (20640,)

print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
# 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)
# :Attribute Information:
#     - MedInc        median income in block group
#     - HouseAge      median house age in block group
#     - AveRooms      average number of rooms per household
#     - AveBedrms     average number of bedrooms per household
#     - Population    block group population
#     - AveOccup      average number of household members
#     - Latitude      block group latitude
#     - Longitude     block group longitude

random_state_value = 7

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.3, random_state = random_state_value)

model = Sequential()
model.add(Dense(16, input_dim = 8))
model.add(Dense(32))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
start_time = round(time.time(), 2)
model.fit(x_train, y_train, epochs = 15000,
          batch_size = 301)
end_time = round(time.time(), 2)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

# random_state = 7, epochs = 8000, batch_size = 301
# 로스 : 0.5393111109733582
# R2 스코어 : 0.5962883600390119
# 걸린 시간 :  264.67 초

# random_state = 7, epochs = 15000, batch_size = 301
# 로스 : 0.5299805998802185
# R2 스코어 : 0.6032728239318499
# 걸린 시간 :  393.6 초

# random_state = 7, epochs = 30000, batch_size = 100
# 로스 : 0.5251055955886841
# R2 스코어 : 0.606922141794031
# 걸린 시간 :  2336.51 초

print("로스 :", loss)
print("R2 스코어 :", r2)
print("걸린 시간 : ", round(end_time - start_time, 2), "초")
print("Random State:", random_state_value)



# model.add(Dense(16, input_dim = 8))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(64))
# model.add(Dense(8))
# model.add(Dense(1))







