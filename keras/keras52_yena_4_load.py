import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, GRU
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time as tm

path_npy = 'c:/_data/_save_npy/'
x = np.load(path_npy + 'keras52_yena3_x.npy')
y = np.load(path_npy + 'keras52_yena3_y.npy')

x = x.astype(np.float32)
y = y.astype(np.float32)

# print(x.shape, y.shape) # (420539, 12, 14) (420539,)
x = x.reshape(-1,1)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
# print(x.shape) # (70650552, 1)
x = x.reshape(420539,12,14)
# print(x.shape) # (420539, 12, 14)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0, shuffle=False)

#2
model = Sequential()
model.add(GRU(22, input_shape = (12,14), activation='sigmoid'))
model.add(Dense(25, activation ='swish'))
model.add(Dense(28, activation ='swish'))
model.add(Dense(27, activation ='swish'))
model.add(Dense(26, activation ='swish'))
model.add(Dense(15, activation ='swish'))
model.add(Dense(1))

model.summary()

#3
model.compile(loss='mse', optimizer = 'adam')
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', verbose = 1,
                   patience = 100, restore_best_weights=True)

fit_start = tm.time()
model.fit(x_train, y_train, epochs = 1000, batch_size = 20000, validation_split = 0.17,
          verbose = 1, callbacks=[es])
fit_end = tm.time()
fit_time = np.round(fit_end - fit_start, 2)


#4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
# print('split_time',split_time)
print('fit_time', fit_time)
print('loss', loss)
print('mse', mse)
print('r2', r2)

# print(y_test.shape, y_predict.shape)
y_test = y_test.reshape(-1,1)
# print(y_test.shape, y_predict.shape)
## ERROR
submit = pd.DataFrame(np.array([y_test, y_predict]).reshape(-1,2), columns = ['test', 'predict'])
submit.to_csv('c:/_data/kaggle/jena/jena_submit_1.csv', index=False)
#   File "c:\Study\keras\keras52_yena_4_load.py", line 65, in <module>
#     submit = pd.DataFrame(np.array([y_test, y_predict]).reshape(-1,2), columns = ['test', 'predict'])
# ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. 
# The detected shape was (2, 63081) + inhomogeneous part.


# loss 9.458792192162946e-06
# [[79.06234]]

# fit_time 107.28
# loss 1.4996258869359735e-05
# mse 1.4996271185029734e-05
# r2 0.9992335538554432