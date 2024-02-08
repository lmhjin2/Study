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

from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')


for name, algorithm in allAlgorithms:
    try:
        #2 모델
        model = algorithm()
        #3 훈련
        model.fit(x_train, y_train)
        #4 평가
        acc = model.score(x_test, y_test)
        print(name, '의 정답률:', acc)
    except Exception as e:
        # print(name, '에러', e)
        continue



# LinearRegression() acc 0.595058137897748
# KNeighborsRegressor() acc 0.5591255934738976
# DecisionTreeRegressor() acc 0.6077962436386359
# RandomForestRegressor() acc 0.8031261776340614 =========================================================================================

# ARDRegression 의 정답률: 0.5952148573217009
# AdaBoostRegressor 의 정답률: 0.4189919957525875
# BaggingRegressor 의 정답률: 0.7903858534562944
# BayesianRidge 의 정답률: 0.5950134868334327
# DecisionTreeRegressor 의 정답률: 0.5936108735157397
# DummyRegressor 의 정답률: -0.00014678004278678713
# ElasticNet 의 정답률: -0.00014678004278678713
# ElasticNetCV 의 정답률: 0.5267835999580448
# ExtraTreeRegressor 의 정답률: 0.5793538547838499
# ExtraTreesRegressor 의 정답률: 0.8083570601770989
# GammaRegressor 의 정답률: 0.015581617185826846
# GaussianProcessRegressor 의 정답률: -1155929.1659276236
# GradientBoostingRegressor 의 정답률: 0.7792412294597976
# HistGradientBoostingRegressor 의 정답률: 0.8396697405505027
# HuberRegressor 의 정답률: -0.928378328442234
# KNeighborsRegressor 의 정답률: 0.5591255934738976
# KernelRidge 의 정답률: 0.5099539001396443
# Lars 의 정답률: 0.5950581378977478
# LarsCV 의 정답률: 0.5941288010403172
# Lasso 의 정답률: -0.00014678004278678713
# LassoCV 의 정답률: 0.5923576894948204
# LassoLars 의 정답률: -0.00014678004278678713
# LassoLarsCV 의 정답률: 0.5941288010403172
# LassoLarsIC 의 정답률: 0.5950581378977478
# LinearRegression 의 정답률: 0.595058137897748
# LinearSVR 의 정답률: 0.5173294566782183
# MLPRegressor 의 정답률: 0.6674223824571357
# NuSVR 의 정답률: 0.5586951133198885
# OrthogonalMatchingPursuit 의 정답률: 0.4455495441385483
# OrthogonalMatchingPursuitCV 의 정답률: 0.5832456894068173
# PLSRegression 의 정답률: 0.4963688752036097
# PassiveAggressiveRegressor 의 정답률: -0.00018542498193441226
# PoissonRegressor 의 정답률: 0.03345561436069755
# RANSACRegressor 의 정답률: -28.375196857726902
# RandomForestRegressor 의 정답률: 0.8047140618984528
# Ridge 의 정답률: 0.546320418898214
# RidgeCV 의 정답률: 0.5895359844495178
# SGDRegressor 의 정답률: 0.48516532125700973
# SVR 의 정답률: 0.5567853691799818
# TheilSenRegressor 의 정답률: -45.08644927365984
# TransformedTargetRegressor 의 정답률: 0.595058137897748
# TweedieRegressor 의 정답률: 0.015569236306808953


# HistGradientBoostingRegressor 의 정답률: 0.8396697405505027