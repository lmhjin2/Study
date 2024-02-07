import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,mean_absolute_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import LinearSVR

plt.rcParams['font.family'],"Malgun Gothic"
plt.rcParams['axes.unicode_minus']=False

#1 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state=0 )

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
# model = LinearSVR()

# #3
# import datetime
# import time as tm

# start_time = tm.time()

# model.fit(x_train, y_train)

# end_time = tm.time()
# run_time = round(end_time - start_time, 2)

# #4
# r2 = model.score(x_test, y_test)
# y_predict = model.predict(x_test)
# r21 = r2_score(y_test, y_predict)

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# def RMSLE(y_test, y_predict):
#     return np.sqrt(mean_squared_log_error(y_test,y_predict))

# rmse = RMSE(y_test, y_predict)

# # print('RMSLE:', rmsle)
# print('R2:',r2)
# print('R2:',r21)
# print('RMSE:', rmse)
# print("run time:", run_time)
# print('california')
'''
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color = 'red',
         label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], color = 'blue',
         label = 'val_loss', marker = '.')
plt.legend(loc='upper right')
plt.title('캘리포니아 로스')
plt.xlabel('epochs')
plt.ylabel('val_loss')
plt.grid()
plt.show()
'''

# loss: [0.5937129259109497, 0.5937129259109497, 0.05482317879796028, 0.5589895248413086]
# RMSE: 0.7705276989046251
# R2: 0.5519150654481986

# scaler = MinMaxScaler()
# loss: [0.5377083420753479, 0.5377083420753479, 0.052736494690179825, 0.5330697298049927]
# RMSE: 0.733286021331258
# R2: 0.5941826188806718

# scaler = MaxAbsScaler()
# loss: [0.5390101075172424, 0.5390101075172424, 0.052398569881916046, 0.5343481302261353]
# RMSE: 0.7341730965529952
# R2: 0.5932001692557324

# MixMaxScaler = MaxAbsScaler

# LSTM
# loss: [0.5205868482589722, 0.5205868482589722, 0.05125166475772858, 0.5364072918891907]
# RMSE: 0.7215170672223521
# R2: 0.6071045051844135
# run time: 130.16

# Conv1D
# loss: [0.5183747410774231, 0.5183747410774231, 0.05027955770492554, 0.5178951025009155]
# RMSE: 0.7199824158542191
# R2: 0.6087740883364874
# run time: 137.63

# R2: 0.5181862165341713
# R2: 0.5181862165341713
# RMSE: 0.7990016829995291
# run time: 0.02

from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# model_P = Perceptron()
model_L = LinearRegression()
model_K = KNeighborsRegressor()
model_D = DecisionTreeRegressor()
model_R = RandomForestRegressor()

models = [ model_L,  model_K, model_D, model_R]

for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(model, "acc", results)

# LinearRegression() acc 0.595058137897748
# KNeighborsRegressor() acc 0.5591255934738976
# DecisionTreeRegressor() acc 0.6077962436386359
# RandomForestRegressor() acc 0.8031261776340614 =========================================================================================


