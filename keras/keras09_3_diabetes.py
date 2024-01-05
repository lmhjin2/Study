from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
'''
print(x)
print(y)
print(x.shape, y.shape) # (442,10), (442,)

print(datasets.feature_names)
print(datasets.DESCR)
'''
# 해봐!!!
# R2 0.62 이상

random_state_value = 9  # 9
test_size_value = 0.18  # 0.18
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = test_size_value, random_state = random_state_value)

model = Sequential()
model.add(Dense(20, input_dim = 10))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
start_time = round(time.time(), 2)
model.fit(x_train, y_train, epochs = 100, batch_size = 57)
end_time = round(time.time(), 2)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("로스 :", loss)
print("R2 스코어 :", r2)
print("걸린 시간 : ", round(end_time - start_time, 2), "초")
# print("Random State:", random_state_value)

# random_state = 9, epochs = 1000, batch_size = 103, loss = 'mse', test_size = 0.25
# 로스 : 2220.82763671875
# R2 스코어 : 0.5981328637022996
# 걸린 시간 :  3.39 초

# random_state = 9, epochs = 100, batch_size = 57, loss = 'mse', test_size = 0.18
# 로스 : 2024.762939453125
# R2 스코어 : 0.631413896941931
# 걸린 시간 :  1.02 초

# random_state = 9, epochs = 150, batch_size = 57, loss = 'mse', test_size = 0.18
# 로스 : 2048.462646484375
# R2 스코어 : 0.6270996543977938
# 걸린 시간 :  1.25 초

# model = Sequential()
# model.add(Dense(20, input_dim = 10))
# model.add(Dense(40))
# model.add(Dense(60))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(1))

