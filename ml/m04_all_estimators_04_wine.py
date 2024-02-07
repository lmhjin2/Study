import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVR, LinearSVC
#1
datasets = load_wine()
x = datasets.data
y = datasets.target

x = x.astype(np.float32)
y = y.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, train_size = 0.8, random_state = 4 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
# model = LinearSVC(C=100)


#3
# model.fit(x_train, y_train)

# #4
# acc = model.score(x_test, y_test)
# y_predict = model.predict(x_test)
# print('acc:', acc)

# acc = accuracy_score(y_predict, y_test)
# print('accuracy_score :', acc)
# print('wine')
# scaler = StandardScaler()
# loss: 0.018835054710507393
# acc: 1.0
# accuracy_score : 1.0

# StandardScaler

# LSTM
# loss: 0.14838989078998566
# acc: 0.9722222089767456
# accuracy_score : 0.9722222222222222

# Conv1D
# loss: 0.6062490344047546
# acc: 0.9444444179534912
# accuracy_score : 0.9444444444444444

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



# Perceptron() acc 0.9722222222222222 =========================================================================================
# LogisticRegression() acc 1.0
# KNeighborsClassifier() acc 0.9722222222222222 =========================================================================================
# DecisionTreeClassifier() acc 0.9444444444444444
# RandomForestClassifier() acc 1.0

# AdaBoostClassifier 의 정답률: 0.9166666666666666
# BaggingClassifier 의 정답률: 1.0
# BernoulliNB 의 정답률: 0.9166666666666666
# CalibratedClassifierCV 의 정답률: 1.0
# DecisionTreeClassifier 의 정답률: 1.0
# DummyClassifier 의 정답률: 0.3888888888888889
# ExtraTreeClassifier 의 정답률: 0.8055555555555556
# ExtraTreesClassifier 의 정답률: 1.0
# GaussianNB 의 정답률: 1.0
# GaussianProcessClassifier 의 정답률: 0.9444444444444444
# GradientBoostingClassifier 의 정답률: 0.9722222222222222
# HistGradientBoostingClassifier 의 정답률: 0.9722222222222222
# KNeighborsClassifier 의 정답률: 0.9722222222222222
# LabelPropagation 의 정답률: 0.8333333333333334
# LabelSpreading 의 정답률: 0.8333333333333334
# LinearDiscriminantAnalysis 의 정답률: 0.9722222222222222
# LinearSVC 의 정답률: 0.9722222222222222
# LogisticRegression 의 정답률: 1.0
# LogisticRegressionCV 의 정답률: 1.0
# MLPClassifier 의 정답률: 1.0
# NearestCentroid 의 정답률: 0.9444444444444444
# NuSVC 의 정답률: 1.0
# PassiveAggressiveClassifier 의 정답률: 1.0
# Perceptron 의 정답률: 0.9722222222222222
# QuadraticDiscriminantAnalysis 의 정답률: 1.0
# RandomForestClassifier 의 정답률: 0.9722222222222222
# RidgeClassifier 의 정답률: 0.9722222222222222
# RidgeClassifierCV 의 정답률: 1.0
# SGDClassifier 의 정답률: 1.0
# SVC 의 정답률: 1.0