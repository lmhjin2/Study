# 5일분 (720행)을 훈련시켜서 
# 하루뒤 (144행)뒤를 예측
# ex) 내가 721번. 1번은 index니까 2번부터 720번까지 훈련, 865를 예측
# 지금했던건 두시간뒤를예측
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import time as tm

path_jena = 'c:/_data/kaggle/jena/'

dataset = pd.read_csv(path_jena + 'jena_climate_2009_2016.csv', index_col=0)
# print(dataset.shape)    # (420551, 14)

# degC 섭씨 column 맨 오른쪽으로 보내기
degC = 'T (degC)'
moved_column = dataset.pop(degC)
dataset[degC] = moved_column
# print(dataset)
# 720+144+1 = 865
timestep = 720    # train size

def split_x(dataset, timestep, column):
    x_arr = []
    y_arr = []
    for i in range(dataset.shape[0]-timestep-144):
        x_subset = dataset.iloc[(i):(i+timestep), : ]
        y_subset = dataset.iloc[(i+timestep+144), column]
        x_arr.append(x_subset)
        y_arr.append(y_subset)
    return np.array(x_arr), np.array(y_arr)


split_start = tm.time()
x, y = split_x(dataset, timestep, 13)
# print(x.shape, y.shape) # (419687, 720, 14) (419687,)
split_end = tm.time()
split_time = np.round(split_end-split_start, 2)
print('split time', split_time, '초')

path_npy = 'c:/_data/_save_npy/'
np.save(path_npy + 'keras52_yena2_1_x.npy', arr = x)
np.save(path_npy + 'keras52_yena2_1_y.npy', arr = y)

# print(y.shape)    # 866 ~ 420552(끝) 까지
