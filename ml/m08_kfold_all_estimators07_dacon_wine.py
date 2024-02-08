# https://dacon.io/competitions/open/235610/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators

path = "c:/_data/dacon/wine/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")


train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2
import warnings
warnings.filterwarnings('ignore')

allAlgorithms = all_estimators(type_filter='classifier')

for name, algorithm in allAlgorithms:
    try:
        #2 모델
        model = algorithm()
        #3 훈련
        scores = cross_val_score(model, x_train, y_train, cv = kfold)
        print('============', name, '============')
        print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
        acc = accuracy_score(y_test, y_predict)
        print('cross_val_precit acc:', acc)
    except Exception as e:
        # print(name, '에러', e)
        continue


# RandomForestClassifier() acc 0.6727272727272727 

# RandomForestClassifier 의 정답률: 0.6709090909090909

# acc: [0.68727273 0.65090909 0.69272727 0.70909091 0.65636364 0.68363636
#  0.68       0.69945355 0.71584699 0.66848816]
#  평균 acc: 0.6844

# acc: [0.69454545 0.66363636 0.66727273 0.7        0.67090909 0.73272727
#  0.68181818 0.63934426 0.67395264 0.67395264]
#  평균 acc: 0.6798

# acc: [0.68545455 0.66909091 0.67090909 0.69636364 0.68727273 0.71818182
#  0.69454545 0.63752277 0.67577413 0.67395264]
#  평균 acc: 0.6809
#  cross_val_precit acc: 0.5809090909090909