# https://dacon.io/competitions/open/235610/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR

path = "c:/_data/dacon/wine/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")


train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, 
                                    train_size = 0.8, random_state = 0 )
   
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2
# model = LinearSVC()

# #3
# # import datetime

# # date = datetime.datetime.now()
# # date = date.strftime("%m%d_%H%M")   # 월일_시분

# # path1 = "c:/_data/_save/MCP/k28/10/"
# # filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
# # filepath = "".join([path1, 'k28_', date, '_', filename])

# start_time = tm.time()

# model.fit(x_train, y_train)

# end_time = tm.time()
# run_time = round(end_time - start_time, 2)

# #4 
# acc = model.score(x_test, y_test)
# y_submit = model.predict(test_csv)
# y_predict = model.predict(x_test)

# submission_csv['quality'] = y_submit
# # submission_csv.to_csv(path + "submission_0116_3.csv", index=False)

# accuracy = accuracy_score(y_predict, y_test) 

# print('acc:', acc)
# print('accuracy_score :', accuracy)
# print("run time:", run_time)
# print('dacon wine')

# scaler = MaxAbsScaler()
# loss: 1.1051563024520874
# acc: 0.5363636612892151
# accuracy_score : 0.5363636363636364

# LSTM
# loss: 1.1040036678314216
# acc: 0.5363636612892151
# accuracy_score : 0.5363636363636364
# run time: 23.48

# Conv1D
# loss: 1.0907200574874878
# acc: 0.5381818413734436
# accuracy_score : 0.5381818181818182
# run time: 27.18

# acc: 0.5172727272727272
# accuracy_score : 0.5172727272727272
# run time: 0.07
# dacon wine

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



# Perceptron() acc 0.4818181818181818
# LogisticRegression() acc 0.5236363636363637
# KNeighborsClassifier() acc 0.5463636363636364
# DecisionTreeClassifier() acc 0.5918181818181818
# RandomForestClassifier() acc 0.6727272727272727 =========================================================================================

# AdaBoostClassifier 의 정답률: 0.36
# BaggingClassifier 의 정답률: 0.6409090909090909
# BernoulliNB 의 정답률: 0.45
# CalibratedClassifierCV 의 정답률: 0.52
# ComplementNB 의 정답률: 0.4090909090909091
# DecisionTreeClassifier 의 정답률: 0.5872727272727273
# DummyClassifier 의 정답률: 0.44
# ExtraTreeClassifier 의 정답률: 0.5872727272727273
# ExtraTreesClassifier 의 정답률: 0.6654545454545454
# GaussianNB 의 정답률: 0.3481818181818182
# GaussianProcessClassifier 의 정답률: 0.519090909090909
# GradientBoostingClassifier 의 정답률: 0.56
# HistGradientBoostingClassifier 의 정답률: 0.6509090909090909
# KNeighborsClassifier 의 정답률: 0.5463636363636364
# LabelPropagation 의 정답률: 0.47363636363636363
# LabelSpreading 의 정답률: 0.4718181818181818
# LinearDiscriminantAnalysis 의 정답률: 0.5127272727272727
# LinearSVC 의 정답률: 0.5163636363636364
# LogisticRegression 의 정답률: 0.5236363636363637
# LogisticRegressionCV 의 정답률: 0.5318181818181819
# MLPClassifier 의 정답률: 0.5336363636363637
# MultinomialNB 의 정답률: 0.4481818181818182
# NearestCentroid 의 정답률: 0.12272727272727273
# PassiveAggressiveClassifier 의 정답률: 0.5254545454545455
# Perceptron 의 정답률: 0.4818181818181818
# QuadraticDiscriminantAnalysis 의 정답률: 0.4909090909090909
# RandomForestClassifier 의 정답률: 0.6709090909090909
# RidgeClassifier 의 정답률: 0.5163636363636364
# RidgeClassifierCV 의 정답률: 0.5181818181818182
# SGDClassifier 의 정답률: 0.4954545454545455
# SVC 의 정답률: 0.5209090909090909
