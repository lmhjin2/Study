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




# Perceptron() r2 0.015
# LinearRegression() r2 0.6189894726936527
# KNeighborsRegressor() r2 0.6930000271623358
# DecisionTreeRegressor() r2 0.5887680535218736
# RandomForestRegressor() r2 0.8111208077261252 =========================================================================================

# ARDRegression 의 정답률: 0.6160668868581636
# AdaBoostRegressor 의 정답률: 0.5851196141338868
# BaggingRegressor 의 정답률: 0.79347257949636
# BayesianRidge 의 정답률: 0.6198084337205655
# DecisionTreeRegressor 의 정답률: 0.6545612666601947
# DummyRegressor 의 정답률: -0.0032043421571521513
# ElasticNet 의 정답률: 0.6045336231964462
# ElasticNetCV 의 정답률: 0.6220123808576097
# ExtraTreeRegressor 의 정답률: 0.5245042936610653
# ExtraTreesRegressor 의 정답률: 0.8096550039121309
# GammaRegressor 의 정답률: 0.46979436714406686
# GaussianProcessRegressor 의 정답률: 0.42834409881525515
# GradientBoostingRegressor 의 정답률: 0.7844768635135471
# HistGradientBoostingRegressor 의 정답률: 0.8275995871226449
# HuberRegressor 의 정답률: 0.6113970779517137
# KNeighborsRegressor 의 정답률: 0.6930000271623358
# KernelRidge 의 정답률: -1.240448266763369
# Lars 의 정답률: 0.6189894726936525
# LarsCV 의 정답률: 0.6163088594761195
# Lasso 의 정답률: 0.6166371565597257
# LassoCV 의 정답률: 0.6163795715174163
# LassoLars 의 정답률: 0.616636292163002
# LassoLarsCV 의 정답률: 0.6163088594761195
# LassoLarsIC 의 정답률: 0.6175159399400965
# LinearRegression 의 정답률: 0.6189894726936527
# LinearSVR 의 정답률: 0.5788897790810046
# MLPRegressor 의 정답률: 0.5942122866310418
# NuSVR 의 정답률: 0.49528107786208064
# OrthogonalMatchingPursuit 의 정답률: 0.4460393958504264
# OrthogonalMatchingPursuitCV 의 정답률: 0.6033328774691
# PLSRegression 의 정답률: 0.6205505091278405
# PassiveAggressiveRegressor 의 정답률: 0.6104507201054761
# PoissonRegressor 의 정답률: 0.6415398823216183
# RANSACRegressor 의 정답률: 0.4361580511458113
# RandomForestRegressor 의 정답률: 0.795691963016971
# Ridge 의 정답률: 0.6190630356243765
# RidgeCV 의 정답률: 0.6196915260534779
# SGDRegressor 의 정답률: 0.6178780904937129
# SVR 의 정답률: 0.48924026944182797
# TheilSenRegressor 의 정답률: 0.6279585380169566
# TransformedTargetRegressor 의 정답률: 0.6189894726936527
# TweedieRegressor 의 정답률: 0.5695226375541397

