import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,cross_val_predict
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

scaler = MaxAbsScaler()

scaler.fit(x)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2
model = HistGradientBoostingRegressor()

# #3
scores = cross_val_score(model, x, y, cv = kfold)
print('r2:', scores, "\n 평균 r2:", round(np.mean(scores), 4))
y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

r2 = r2_score(y_test, y_predict)
print('cross_val_precit r2:', r2)
#4

# HistGradientBoostingRegressor 의 정답률: 0.8396697405505027

# acc: [0.83959202 0.83979848 0.84652567 0.8575718  0.83906609 0.83876288
#  0.83122632 0.82068198 0.8285659  0.82990476]
#  평균 acc: 0.8372

# acc: [0.84078301 0.85270921 0.84025083 0.81923533 0.85904233 0.85285232
#  0.84569715 0.8378194  0.85035462 0.85126993]
#  평균 acc: 0.845

# r2: [0.84219754 0.85216254 0.8381129  0.8200251  0.85521116 0.85554249
#  0.84646818 0.83683306 0.8463955  0.85043873]
#  평균 r2: 0.8443
# cross_val_precit r2: 0.8157203675530726
