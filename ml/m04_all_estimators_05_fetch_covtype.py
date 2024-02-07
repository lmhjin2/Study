import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, train_size = 0.8, random_state = 0 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2

#3
# st_time = tm.time()
# model.fit(x_train, y_train)
# e_time = tm.time()
# r_time = np.round(e_time - st_time, 2)
# #4
# acc = model.score(x_test, y_test)
# y_predict = model.predict(x_test)

# accuracy = accuracy_score(y_predict, y_test)
# print('acc:', acc)
# print('accuracy:', accuracy)
# print('run time ', r_time)
# print('fetch covtype')

# RobustScaler  # MCP
# loss: 0.19374237954616547
# acc: 0.9310688972473145
# accuracy_score : 0.9310689052778327

# acc: 0.7124686970215915
# accuracy: 0.7124686970215915
# run time  157.25

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



# Perceptron() acc 0.606714112372314
# LogisticRegression() acc 0.7252480572790719
# KNeighborsClassifier() acc 0.9283323150004733
# DecisionTreeClassifier() acc 0.9386332538746848 
# RandomForestClassifier() acc 0.9546397253082967 =========================================================================================

# AdaBoostClassifier 의 정답률: 0.44818980577093537
# BaggingClassifier 의 정답률: 0.9608874125452871
# BernoulliNB 의 정답률: 0.6612651996936396
# CalibratedClassifierCV 의 정답률: 0.7130366685885906
# DecisionTreeClassifier 의 정답률: 0.9384267187594124
# DummyClassifier 의 정답률: 0.48760359026875383
# ExtraTreeClassifier 의 정답률: 0.8757605225338416
# ExtraTreesClassifier 의 정답률: 0.9538652186260251
# GaussianNB 의 정답률: 0.09089266197946697
# GradientBoostingClassifier 의 정답률: 0.7722864297823636
# HistGradientBoostingClassifier 의 정답률: 0.7780952298994002
# KNeighborsClassifier 의 정답률: 0.9283323150004733
# LinearDiscriminantAnalysis 의 정답률: 0.6794919236164298
# LinearSVC 의 정답률: 0.7124686970215915
# LogisticRegression 의 정답률: 0.7252480572790719
# LogisticRegressionCV 의 정답률: 0.7253082966876931
# MLPClassifier 의 정답률: 0.8773439584175968
# NearestCentroid 의 정답률: 0.44946343898178187
# PassiveAggressiveClassifier 의 정답률: 0.5666893281584813
# Perceptron 의 정답률: 0.606714112372314
# QuadraticDiscriminantAnalysis 의 정답률: 0.08509246749223343
# RandomForestClassifier 의 정답률: 0.9549581336110083
# RidgeClassifier 의 정답률: 0.7009199418259425
# RidgeClassifierCV 의 정답률: 0.7009113361961395
# SGDClassifier 의 정답률: 0.7103861346092614
# SVC 의 정답률: 0.8334380351626034

