import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

#1
datasets = load_iris()
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify= y, train_size = 0.8, random_state = 0 )

#2
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
    # print(allAlgorithms)
    # print(len(allAlgorithms)) # classifier: 41 / regressor: 55

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
    

# AdaBoostClassifier 의 정답률: 0.9333333333333333
# BaggingClassifier 의 정답률: 0.9333333333333333
# BernoulliNB 의 정답률: 0.3333333333333333
# CalibratedClassifierCV 의 정답률: 0.9333333333333333
# CategoricalNB 의 정답률: 0.9
# ComplementNB 의 정답률: 0.6666666666666666
# DecisionTreeClassifier 의 정답률: 0.9666666666666667
# DummyClassifier 의 정답률: 0.3333333333333333
# ExtraTreeClassifier 의 정답률: 1.0
# ExtraTreesClassifier 의 정답률: 1.0
# GaussianNB 의 정답률: 0.9666666666666667
# GaussianProcessClassifier 의 정답률: 1.0
# GradientBoostingClassifier 의 정답률: 0.9666666666666667
# HistGradientBoostingClassifier 의 정답률: 0.9333333333333333
# KNeighborsClassifier 의 정답률: 1.0
# LabelPropagation 의 정답률: 0.9666666666666667
# LabelSpreading 의 정답률: 0.9666666666666667
# LinearDiscriminantAnalysis 의 정답률: 1.0
# LinearSVC 의 정답률: 0.9666666666666667
# LogisticRegression 의 정답률: 1.0
# LogisticRegressionCV 의 정답률: 1.0
# MLPClassifier 의 정답률: 1.0
# MultinomialNB 의 정답률: 0.9666666666666667
# NearestCentroid 의 정답률: 0.8666666666666667
# NuSVC 의 정답률: 1.0
# PassiveAggressiveClassifier 의 정답률: 0.7333333333333333
# Perceptron 의 정답률: 0.9666666666666667
# QuadraticDiscriminantAnalysis 의 정답률: 1.0
# RadiusNeighborsClassifier 의 정답률: 0.9666666666666667
# RandomForestClassifier 의 정답률: 0.9333333333333333
# RidgeClassifier 의 정답률: 0.8333333333333334
# RidgeClassifierCV 의 정답률: 0.8333333333333334
# SGDClassifier 의 정답률: 0.9666666666666667
# SVC 의 정답률: 1.0


# TypeError: __init__() missing 1 required positional argument: 'base_estimator'
# argument = 하이퍼 파라미터

# GaussianProcessClassifier 의 정답률: 1.0
# KNeighborsClassifier 의 정답률: 1.0
# LinearDiscriminantAnalysis 의 정답률: 1.0
# LogisticRegression 의 정답률: 1.0
# LogisticRegressionCV 의 정답률: 1.0
# MLPClassifier 의 정답률: 1.0
# NuSVC 의 정답률: 1.0
# QuadraticDiscriminantAnalysis 의 정답률: 1.0
# SVC 의 정답률: 1.0

'''
#3
model.fit(x_train, y_train)

#4
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)

# print(y_predict)
acc = accuracy_score(y_predict, y_test)

print('model.score :', results)
print("accuracy_score :", acc)
'''
