import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# x = x.reshape(569, 6,5,1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state = 1 )

#2 
# model = Perceptron()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

#3
# model.fit(x_train, y_train)
# # loss: 0.09776605665683746
# # r2: 0.5982901288073281

# #4 
# results = model.score(x_test, y_test)   # x_test를 넣어서 predict한 값을 y_test 와 비교.
# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)


# print('acc:', results)
# print('r2:', r2)

# scaler = RobustScaler()
# loss: [0.07344205677509308, 0.9651162624359131, 0.02196618542075157, 0.054660264402627945]
# acc:  0.9651162790697675
# r2: 0.9081097883593787

# acc: 0.9418604651162791
# r2: 0.7567873303167421
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




# model = Perceptron()
# acc: 0.7209302325581395
# r2: -0.16742081447963786

# model = LogisticRegression() =========================================================================================
# acc: 0.9534883720930233
# r2: 0.8054298642533937

# model = KNeighborsClassifier()
# acc: 0.9418604651162791
# r2: 0.7567873303167421

# model = DecisionTreeClassifier() =========================================================================================
# acc: 0.9534883720930233
# r2: 0.8054298642533937

# model = RandomForestClassifier()
# acc: 0.9302325581395349
# r2: 0.7081447963800905

## LogisticRegression = DecisionTreeClassifier

# AdaBoostClassifier 의 정답률: 0.9302325581395349
# BaggingClassifier 의 정답률: 0.9651162790697675
# BernoulliNB 의 정답률: 0.6046511627906976
# CalibratedClassifierCV 의 정답률: 0.8837209302325582
# ComplementNB 의 정답률: 0.872093023255814
# DecisionTreeClassifier 의 정답률: 0.9534883720930233
# DummyClassifier 의 정답률: 0.6046511627906976
# ExtraTreeClassifier 의 정답률: 0.9302325581395349
# ExtraTreesClassifier 의 정답률: 0.9418604651162791
# GaussianNB 의 정답률: 0.9418604651162791
# GaussianProcessClassifier 의 정답률: 0.9534883720930233
# GradientBoostingClassifier 의 정답률: 0.9534883720930233
# HistGradientBoostingClassifier 의 정답률: 0.9534883720930233
# KNeighborsClassifier 의 정답률: 0.9418604651162791
# LabelPropagation 의 정답률: 0.43023255813953487
# LabelSpreading 의 정답률: 0.43023255813953487
# LinearDiscriminantAnalysis 의 정답률: 0.9302325581395349
# LinearSVC 의 정답률: 0.872093023255814
# LogisticRegression 의 정답률: 0.9534883720930233
# LogisticRegressionCV 의 정답률: 0.9534883720930233
# MLPClassifier 의 정답률: 0.9186046511627907
# MultinomialNB 의 정답률: 0.872093023255814
# NearestCentroid 의 정답률: 0.872093023255814
# NuSVC 의 정답률: 0.8604651162790697
# PassiveAggressiveClassifier 의 정답률: 0.9302325581395349
# Perceptron 의 정답률: 0.7209302325581395
# QuadraticDiscriminantAnalysis 의 정답률: 0.9418604651162791
# RandomForestClassifier 의 정답률: 0.9302325581395349
# RidgeClassifier 의 정답률: 0.9186046511627907
# RidgeClassifierCV 의 정답률: 0.9186046511627907
# SGDClassifier 의 정답률: 0.7906976744186046
# SVC 의 정답률: 0.9069767441860465


# BaggingClassifier 의 정답률: 0.9651162790697675