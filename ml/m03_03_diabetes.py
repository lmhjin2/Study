import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_squared_log_error
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state = 0 )

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# # scaler = MinMaxScaler()
# scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# # print(x_train.shape, x_test.shape)

#2
# model = LinearSVR(C=1000)

# #3
# model.fit(x_train, y_train)

# #4
# r2 = model.score(x_test,y_test)

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# def RMSLE(y_test, y_predict):
#     return np.sqrt(mean_squared_log_error(y_test, y_predict))

# rmse = RMSE(y_test, y_predict)
# rmsle = RMSLE(y_test, y_predict)

# print('r2:' , r2)
# print('RMSE:', rmse)
# print('RMSLE:', rmsle)
# print('diabetes')

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

# scaler = StandardScaler()
# loss: [3690.262451171875, 3690.262451171875, 0.17990046739578247, 47.92462921142578]
# RMSE: 60.74752928544592
# RMSLE: 0.4241467632371064

# r2: 0.19037582699251843
# RMSE: 62.21322867804605
# RMSLE: 0.42893657582040895

# StandardScaler

# r2: 0.20390275334794683
# RMSE: 61.691320758037534
# RMSLE: 0.42650359148427125
# diabetes

# Perceptron() r2 0.014925373134328358
# LinearRegression() r2 0.2303903470818217 =========================================================================================
# KNeighborsRegressor() r2 -0.0004412413104994961
# DecisionTreeRegressor() r2 -0.884614950894437
# RandomForestRegressor() r2 0.1295744088399613
