# restore_best_weights 와
# save_best_only 에 대한 고찰

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVC, LinearSVR

#1
datasets = load_boston()

x = datasets.data
y = datasets.target

random_state_value = 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = random_state_value )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2
# model = LinearSVR()

# #3
# import time as tm
# start_time = tm.time()

# model.fit(x_train, y_train)

# end_time = tm.time()
# run_time = round(end_time - start_time, 2)

# #4
# r2 = model.score(x_test, y_test)

# y_predict = model.predict(x_test)
# r21 = r2_score(y_test, y_predict)
# print("R2 스코어 :", r2)
# print("R2 스코어 :", r21)

# print("run time:", run_time)
# print('boston')

# Conv1D
# 로스 : 3.76533579826355
# R2 스코어 : 0.6851054006763155
# run time: 5.88

# R2 스코어 : 0.744220072538414
# R2 스코어 : 0.744220072538414
# run time: 0.0

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


# LinearRegression() acc 0.7634174432138483
# KNeighborsRegressor() acc 0.5401612153026705
# DecisionTreeRegressor() acc 0.7268459868747735
# RandomForestRegressor() acc 0.9132005612253553 =========================================================================================

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



