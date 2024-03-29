import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,mean_absolute_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

plt.rcParams['font.family'],"Malgun Gothic"
plt.rcParams['axes.unicode_minus']=False

#1 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# 라벨 인코딩. StratifiedKFold 할때만 필요
label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)
# 라벨 인코딩. StratifiedKFold 할때만 필요

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
scaler = MaxAbsScaler()

scaler.fit(x)
x = scaler.transform(x)

#2
model = HistGradientBoostingRegressor()

# #3
scores = cross_val_score(model, x, y, cv = kfold)
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))

#4

# HistGradientBoostingRegressor 의 정답률: 0.8396697405505027

# acc: [0.83959202 0.83979848 0.84652567 0.8575718  0.83906609 0.83876288
#  0.83122632 0.82068198 0.8285659  0.82990476]
#  평균 acc: 0.8372

# acc: [0.84078301 0.85270921 0.84025083 0.81923533 0.85904233 0.85285232
#  0.84569715 0.8378194  0.85035462 0.85126993]
#  평균 acc: 0.845