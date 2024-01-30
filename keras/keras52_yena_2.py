# 소스 + 결과 메일 제출
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Conv2D, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import time as tm

# x = (n,24,14) / y = (n,)

data = pd.read_csv('c:/_data/kaggle/jena/jena_climate_2009_2016.csv', index_col=0)
size = 144
# print(data.columns)
col = data.columns

scaler = MinMaxScaler()
scaler_y = MinMaxScaler().fit(np.array(data['T (degC)']).reshape(-1,1))
data = scaler.fit_transform(data)
data = pd.DataFrame(data, columns=col)

def split_xy(data, time_step, y_col):
    result_x = []
    result_y = []
    
    num = len(data) - time_step
    for i in range(num):
        result_x.append(data[i:i+time_step])
        y_row = data.iloc[i+time_step]
        result_y.append(y_row[y_col])
    
    return np.array(result_x), np.array(result_y)

x, y = split_xy(data,30,'T (degC)')

path_npy = 'c:/_data/_save_npy/'
np.save(path_npy + 'keras52_yena_x.npy', arr = x)
np.save(path_npy + 'keras52_yena_y.npy', arr = y)

path_npy = 'c:/_data/_save_npy/'
x = np.load(path_npy + 'keras52_yena_x.npy')
y = np.load(path_npy + 'keras52_yena_y.npy')



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.18, random_state=0)

#2
model = Sequential()
model.add(GRU(22, input_shape = (30,14), activation='sigmoid'))
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
model.fit(x_train, y_train, epochs = 1000, batch_size = 20000, validation_split = 0.2,
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

predicted_degC = scaler_y.inverse_transform(np.array(y_predict).reshape(-1,1))
y_true = scaler_y.inverse_transform(np.array(y_test).reshape(-1,1))

submit = pd.DataFrame(np.array([y_true, predicted_degC]).reshape(-1,2), columns = ['true', 'predict'])
submit.to_csv('c:/_data/kaggle/jena/jena_submit.csv', index=False)




# fit_time 905.69
# loss 1.2866002180089708e-05
# mse 1.2866009574829116e-05
# r2 0.9993432310315777