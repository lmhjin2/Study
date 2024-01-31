import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, GRU
from keras.callbacks import EarlyStopping
#1
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10,],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])    
# 80 뽑아라
# print(x.shape, y.shape) # (13, 3) (13,)
x = x.reshape(13,3,1)
x_predict = x_predict.reshape(-1,3,1)

#2 
model = Sequential()
model.add(Conv1D(26, 2, input_shape = (3,1)))
model.add(LSTM(15))
model.add(Dense(32))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(1))
model.summary()
#3
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode= 'auto', patience = 200,
                   verbose = 1, restore_best_weights=True)
model.fit(x, y, epochs = 10000, callbacks=[es])
#4
loss = model.evaluate(x, y)
y_predict = model.predict(x_predict)
print('loss', loss)
print(y_predict)
# loss 8.369144052267075e-05
# [[78.91469]]

# 80근처.
# loss 0.0006026474293321371
# [[77.05156]]
# loss 5.97670950810425e-05
# [[78.03475]]
# loss 3.1954896257957444e-05
# [[77.91103]]



