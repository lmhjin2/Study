import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVR, LinearSVC, SVC

#1
datasets = load_wine()
x = datasets.data
y = datasets.target

x = x.astype(np.float32)
y = y.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2
model = SVC()


#3
scores = cross_val_score(model, x, y, cv = kfold)
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
#4
y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

acc = accuracy_score(y_test, y_predict)
print('cross_val_precit acc:', acc)

# BaggingClassifier 의 정답률: 1.0
# CalibratedClassifierCV 의 정답률: 1.0
# DecisionTreeClassifier 의 정답률: 1.0
# ExtraTreesClassifier 의 정답률: 1.0
# GaussianNB 의 정답률: 1.0
# LogisticRegression 의 정답률: 1.0
# LogisticRegressionCV 의 정답률: 1.0
# MLPClassifier 의 정답률: 1.0
# NuSVC 의 정답률: 1.0
# PassiveAggressiveClassifier 의 정답률: 1.0
# QuadraticDiscriminantAnalysis 의 정답률: 1.0
# RidgeClassifierCV 의 정답률: 1.0
# SGDClassifier 의 정답률: 1.0
# SVC 의 정답률: 1.0

# acc: [1.         1.         1.         0.94444444 1.         1.
#  0.94444444 1.         1.         0.94117647]
#  평균 acc: 0.983

# acc: [1.         1.         1.         1.         0.94444444 1.
#  0.94444444 1.         1.         0.94117647]
#  평균 acc: 0.983


# acc: [0.72222222 0.72222222 0.77777778 0.77777778 0.61111111 0.5
#  0.5        0.72222222 0.58823529 0.76470588]
#  평균 acc: 0.6686
# cross_val_precit acc: 1.0