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


# ARDRegression 의 정답률: 0.22491743194397085
# AdaBoostRegressor 의 정답률: 0.15741507916284703
# BaggingRegressor 의 정답률: -0.029544995935246288
# BayesianRidge 의 정답률: 0.22831048250534458
# DecisionTreeRegressor 의 정답률: -0.7465349872186022
# DummyRegressor 의 정답률: -0.016252639038849326
# ElasticNet 의 정답률: -0.00936087906821359
# ElasticNetCV 의 정답률: 0.26655983401648775
# ExtraTreeRegressor 의 정답률: -0.6292261813951048
# ExtraTreesRegressor 의 정답률: 0.11694861074233875
# GammaRegressor 의 정답률: -0.010940747245726357
# GaussianProcessRegressor 의 정답률: -8.239385501471784
# GradientBoostingRegressor 의 정답률: 0.03499382152006847
# HistGradientBoostingRegressor 의 정답률: -0.05047257084193579
# HuberRegressor 의 정답률: 0.20451228535849109
# KNeighborsRegressor 의 정답률: -0.0004412413104994961
# KernelRidge 의 정답률: -5.040780187881607
# Lars 의 정답률: 0.2566428085584206
# LarsCV 의 정답률: 0.23845589686316793
# Lasso 의 정답률: 0.20570023597455667
# LassoCV 의 정답률: 0.22695123905554315
# LassoLars 의 정답률: 0.2096523550864251
# LassoLarsCV 의 정답률: 0.22517044372320838
# LassoLarsIC 의 정답률: 0.23288853746366622
# LinearRegression 의 정답률: 0.2303903470818217
# LinearSVR 의 정답률: -0.652952975557012
# MLPRegressor 의 정답률: -4.28909819664153
# NuSVR 의 정답률: 0.11534676842068137
# OrthogonalMatchingPursuit 의 정답률: 0.15682469350172068
# OrthogonalMatchingPursuitCV 의 정답률: 0.20960931142668127
# PLSRegression 의 정답률: 0.2200716573501722
# PassiveAggressiveRegressor 의 정답률: 0.2598684542596338
# PoissonRegressor 의 정답률: 0.2197497643442633
# QuantileRegressor 의 정답률: -0.08887011403812739
# RANSACRegressor 의 정답률: -0.4177901082347677
# RadiusNeighborsRegressor 의 정답률: -0.016252639038849326
# RandomForestRegressor 의 정답률: 0.11815289706093568
# Ridge 의 정답률: 0.26373976972994195
# RidgeCV 의 정답률: 0.24155597664040895
# SGDRegressor 의 정답률: 0.2595892511051556
# SVR 의 정답률: 0.06584215312739616
# TheilSenRegressor 의 정답률: 0.19025531678382046
# TransformedTargetRegressor 의 정답률: 0.2303903470818217
# TweedieRegressor 의 정답률: -0.01119102243814285

# ElasticNetCV 의 정답률: 0.26655983401648775