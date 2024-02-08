# https://dacon.io/competitions/open/236070/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators

#1
path = "c:/_data/dacon/iris/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv.drop(['species'], axis = 1)
y = train_csv['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2
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
# acc: 1.0
# accuracy_score : 1.0


# CalibratedClassifierCV 의 정답률: 1.0
# GaussianProcessClassifier 의 정답률: 1.0
# LinearDiscriminantAnalysis 의 정답률: 1.0
# LinearSVC 의 정답률: 1.0
# LogisticRegression 의 정답률: 1.0

# acc: [0.83333333 1.         1.         0.83333333 1.         0.83333333
#  1.         1.         1.         1.        ]
#  평균 acc: 0.95

# acc: [1.         0.91666667 1.         1.         1.         0.83333333
#  0.91666667 1.         0.91666667 1.        ]
#  평균 acc: 0.9583

# acc: [1.         0.91666667 1.         1.         1.         0.83333333
#  0.91666667 1.         0.91666667 1.        ]
#  평균 acc: 0.9583