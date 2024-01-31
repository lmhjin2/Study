import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random, datetime
from sklearn.preprocessing import MinMaxScaler

path = 'c:/_data/kaggle/jena/'

dataset = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
print(dataset.shape)  # (420551, 15)  # index_col = 0(420551, 14)

# target_column = 'T (degC)'
# moved_column_df = dataset.pop(target_column) # 안되면 drop -> pop
# dataset[target_column] = moved_column_df

# size = 12

# def split_x(dataset, size, column):
#     x_arr = []
#     y_arr = []
#     for i in range(dataset.shape[0]-size):
#         x_subset = dataset.iloc[i:(i+size), :]  # i 부터 i+size-1행까지
#         y_subset = dataset.iloc[(i+size), column] # 위 x의 타겟값 y 지정
#         x_arr.append(x_subset)
#         y_arr.append(y_subset)
#     return np.array(x_arr), np.array(y_arr)

# x, y = split_x(dataset,size,13)

# # print(x.shape) # (420539, 12, 14)

# path_npy = 'c:/_data/_save_npy/'
# np.save(path_npy + 'keras52_yena3_x.npy', arr = x)
# np.save(path_npy + 'keras52_yena3_y.npy', arr = y)


# print(y)