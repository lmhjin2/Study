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

# print(data.shape)   # (420551, 14)
# ---------------------------------------------------------------------------------------------
# def split_x(dataset,size):
#     aaa = []
#     for i in range(len(dataset) - size + 1):
#         aaa.append(dataset[i : (i+size)])
#     return np.array(aaa)

# # split_start = tm.time()
# bbb = split_x(data,size)

# x = bbb[:,:]    # [행,열]
# y = bbb[1:3,1:3]
# # split_end = tm.time()
# # split_time = np.round(split_end - split_start, 2)
# ---------------------------------------------------------------------------------------------
# features = ["p (mbar)","T (degC)","Tpot (K)","Tdew (degC)","rh (%)",
#             "VPmax (mbar)","VPact (mbar)","VPdef (mbar)","sh (g/kg)","H2OC (mmol/mol)",
#             "rho (g/m**3)","wv (m/s)","max. wv (m/s)","wd (deg)"]

# x = data[features].astype(np.float32)
# y = data["T (degC)"].astype(np.float32)

# x_timesteps = []
# y_timesteps = []

# for i in range(len(x)-size):
#     x_timesteps.append(x.iloc[i:i+size].values)
#     y_timesteps.append(y.iloc[i+size])

# x_timesteps = np.asarray(x_timesteps)
# y_timesteps = np.asarray(y_timesteps)
# # print(x_timesteps.shape, y_timesteps.shape) # (420407, 144, 14) (420407,)
# # ---------------------------------------------------------------------------------------------
# path_npy = 'c:/_data/_save_npy/'
# np.save(path_npy + 'keras52_yena_x.npy', arr = x)
# np.save(path_npy + 'keras52_yena_y.npy', arr = y)
# np.save(path_npy + 'keras52_yena_x_timesteps.npy', arr = x_timesteps)
# np.save(path_npy + 'keras52_yena_y_timesteps.npy', arr = y_timesteps)
# print(x.shape, y.shape) # (420408, 144, 14) (420408, 14)
# ---------------------------------------------------------------------------------------------
path_npy = 'c:/_data/_save_npy/'
x = np.load(path_npy + 'keras52_yena_x.npy')
y = np.load(path_npy + 'keras52_yena_y.npy')
x_timesteps = np.load(path_npy + 'keras52_yena_x_timesteps.npy')
y_timesteps = np.load(path_npy + 'keras52_yena_y_timesteps.npy')
# print(x.shape, y.shape) # (420408, 143, 14) (420408, 14)
# ---------------------------------------------------------------------------------------------
# x = x.reshape(420408, 144*14)
# x_timesteps = x_timesteps.reshape(420408, 143*14)

# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)

# x = x.reshape(-1, 144, 14)
# x_timesteps = x_timesteps.reshape(-1, 144, 14)

x_train, x_test, y_train, y_test = train_test_split(x_timesteps, y_timesteps, 
                                                    test_size=0.18, random_state=0)

#2
model = Sequential()
model.add(GRU(2, input_shape = (144,14), activation='sigmoid'))
model.add(Dense(5, activation ='swish'))
model.add(Dense(8, activation ='swish'))
model.add(Dense(7, activation ='swish'))
model.add(Dense(6, activation ='swish'))
model.add(Dense(5, activation ='swish'))
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

# loss 9.458792192162946e-06
# [[79.06234]]

