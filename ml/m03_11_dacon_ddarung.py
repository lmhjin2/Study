# https://dacon.io/competitions/open/235576/mysubmission
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error, mean_absolute_error
from sklearn.svm import LinearSVR

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.dropna()  # 결측치 드랍.
test_csv = test_csv.fillna(test_csv.mean()) # 결측치에 평균치넣기
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, random_state=151717 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2
model = LinearSVR()

#3
# import time as tm

# start_time = tm.time()

# model.fit(x_train, y_train)

# end_time = tm.time()
# run_time = round(end_time - start_time, 2)

# #4
# r2 = model.score(x_test, y_test)
# y_submit = model.predict(test_csv)
# y_predict = model.predict(x_test)
# r2s = r2_score(y_test, y_predict)

# submission_csv['count']=y_submit
# submission_csv.to_csv(path+"submission_0131.csv",index=False)

# print('r2:', r2)
# print('r2s:', r2s)
# print("run time:", run_time)
# print('따릉')

# loss: [2494.04052734375, 2494.04052734375, 0.6432426571846008, 39.16857147216797]
# RMSE: 49.94036818182851
# r2: 0.6079631835073899


# scaler = MinMaxScaler()
# loss: [2403.452392578125, 2403.452392578125, 0.5717068910598755, 37.98446273803711]
# RMSE: 49.02501764490937
# r2: 0.6222026637380059

# scaler = StandardScaler()
# loss: [2379.362548828125, 2379.362548828125, 0.5877285003662109, 36.955265045166016]
# RMSE: 48.77871023779605
# r2: 0.6259893232430387

# scaler = MaxAbsScaler()
# loss: [2399.464111328125, 2399.464111328125, 0.5455405712127686, 37.7288703918457]
# RMSE: 48.984325076426565
# r2: 0.6228295748300605

# StandardScaler

# LSTM
# loss: [1889.5020751953125, 1889.5020751953125, 0.22933264076709747, 31.007349014282227]
# RMSE: 43.46840188385031
# RMSLE: 0.4788868683730899
# r2: 0.702990239551693
# run time: 135.62

# r2: 0.578639598507783
# r2s: 0.578639598507783

from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model_P = Perceptron()
model_L = LinearRegression()
model_K = KNeighborsRegressor()
model_D = DecisionTreeRegressor()
model_R = RandomForestRegressor()

models = [model_P, model_L,  model_K, model_D, model_R]

for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(model, "r2", results)


# Perceptron() r2 0.015
# LinearRegression() r2 0.6189894726936527
# KNeighborsRegressor() r2 0.6930000271623358
# DecisionTreeRegressor() r2 0.5887680535218736
# RandomForestRegressor() r2 0.8111208077261252 =========================================================================================