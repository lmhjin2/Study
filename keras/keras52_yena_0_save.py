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

x, y = split_xy(data,144,'T (degC)')

path_npy = 'c:/_data/_save_npy/'
np.save(path_npy + 'keras52_yena_x.npy', arr = x)
np.save(path_npy + 'keras52_yena_y.npy', arr = y)