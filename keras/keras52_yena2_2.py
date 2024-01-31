import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time as tm

path_npy = 'c:/_data/_save_npy/'
x = np.load(path_npy + 'keras52_yena2_1_x.npy')
y = np.load(path_npy + 'keras52_yena2_1_y.npy')

x = x.astype(np.float32)
y = y.astype(np.float32)

# print(x.shape, y.shape) # (419687, 720, 14) (419687,)
x = x.reshape(-1,1)
# print(x.shape)  # (419687*720*14, 1) = (4230444960, 1)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x = x.reshape(419687, 720, 14)
# print(x.shape)  # (419687, 720, 14)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0, shuffle=False)

#2
model=Sequential()
model.add(GRU(2, input_shape=(720, 14), activation='sigmoid'))
model.add(Dense(1))

model.summary()

#3
model.compile(loss = 'mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode = 'min',verbose=1,
                   patience=100, restore_best_weights=True)

fit_start = tm.time()
model.fit(x_train, y_train, epochs = 1, batch_size = 2000,
          validation_split=0.2, callbacks=[es])
fit_end = tm.time()
fit_time = np.round(fit_end-fit_start, 2)

#4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('fit_time', fit_time, '초')
print('loss', loss)
print('mse', mse)
print('r2', r2)

print(y_test.shape, y_predict.shape)    # (n,) (n,1)
y_test = y_test.reshape(-1,1)

submit = pd.DataFrame(np.array([y_test, y_predict]).reshape(-1,2), columns = ['test', 'predict'])
submit.to_csv('c:/_data/kaggle/jena/jena2_submit_1.csv', index=False)