import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
from keras.callbacks import EarlyStopping

#1
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape) # (7, 3) (7,)
x = x.reshape(7, 3, 1)
print(x.shape, y.shape) # (7, 3, 1) (7,)

#2
model = Sequential()
# model.add(SimpleRNN(units = 10, input_shape = (3,1)))   # timesteps, features
# 3-D tensor with shape (batch_size, timesteps, features).
model.add(SimpleRNN(10, input_length=3, input_dim = 1)) # input_length = timesteps / input_dim = features
# model.add(SimpleRNN(10, input_dim = 1, input_length=3)) # 순서 바꿔도 되기는함. 가독성이 떨어질뿐
model.add(Dense(117, activation='relu'))
model.add(Dense(213, activation='relu'))
model.add(Dense(117, activation='relu'))
model.add(Dense(214, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(1))


#3
es = EarlyStopping(monitor='loss', mode = 'auto',
                   patience = 200, verbose = 1,
                   restore_best_weights = True)
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs = 10000, callbacks=[es])

#4
results = model.evaluate(x, y)
print('loss', results)
y_pred = np.array([8,9,10]).reshape(1,3,1)
y_pred = model.predict(y_pred)

print('[8,9,10]의 결과', y_pred)
