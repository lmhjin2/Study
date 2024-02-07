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

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')


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


