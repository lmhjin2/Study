import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.utils import all_estimators

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


# ============ AdaBoostClassifier ============
# acc: [0.93333333 0.93333333 0.78571429 1.         1.         0.71428571
#  1.         1.         0.85714286 0.64285714]
#  평균 acc: 0.8867
# cross_val_precit acc: 0.9166666666666666
# ============ BaggingClassifier ============
# acc: [1.         0.86666667 0.92857143 1.         1.         1.
#  0.92857143 1.         0.85714286 1.        ]
#  평균 acc: 0.9581
# cross_val_precit acc: 0.9722222222222222
# ============ BernoulliNB ============
# acc: [1.         0.86666667 0.92857143 0.92857143 0.85714286 0.92857143
#  0.85714286 1.         0.92857143 0.78571429]
#  평균 acc: 0.9081
# cross_val_precit acc: 0.9166666666666666
# ============ CalibratedClassifierCV ============
# acc: [1.         1.         1.         0.92857143 0.92857143 0.92857143
#  1.         1.         0.92857143 0.92857143]
#  평균 acc: 0.9643
# cross_val_precit acc: 1.0
# ============ DecisionTreeClassifier ============
# acc: [1.         0.86666667 0.85714286 0.92857143 0.92857143 0.92857143
#  1.         1.         0.64285714 0.85714286]
#  평균 acc: 0.901
# cross_val_precit acc: 0.9166666666666666
# ============ DummyClassifier ============
# acc: [0.4        0.4        0.42857143 0.42857143 0.42857143 0.42857143
#  0.42857143 0.35714286 0.35714286 0.35714286]
#  평균 acc: 0.4014
# cross_val_precit acc: 0.3888888888888889
# ============ ExtraTreeClassifier ============
# acc: [1.         0.73333333 0.92857143 0.92857143 0.85714286 0.71428571
#  0.78571429 0.92857143 0.85714286 0.92857143]
#  평균 acc: 0.8662
# cross_val_precit acc: 0.8888888888888888
# ============ ExtraTreesClassifier ============
# acc: [1.         1.         0.92857143 1.         1.         1.
#  0.92857143 1.         1.         1.        ]
#  평균 acc: 0.9857
# cross_val_precit acc: 1.0
# ============ GaussianNB ============
# acc: [1.         0.93333333 0.92857143 1.         1.         1.
#  0.92857143 1.         0.92857143 0.85714286]
#  평균 acc: 0.9576
# cross_val_precit acc: 0.9444444444444444
# ============ GaussianProcessClassifier ============
# acc: [1.         0.86666667 0.92857143 1.         1.         0.92857143
#  0.92857143 1.         0.92857143 0.85714286]
#  평균 acc: 0.9438
# cross_val_precit acc: 0.9722222222222222
# ============ GradientBoostingClassifier ============
# acc: [1.         0.93333333 0.92857143 0.92857143 1.         0.85714286
#  0.85714286 0.92857143 0.92857143 1.        ]
#  평균 acc: 0.9362
# cross_val_precit acc: 0.8888888888888888
# ============ HistGradientBoostingClassifier ============
# acc: [1.         1.         0.92857143 1.         1.         1.
#  0.92857143 0.92857143 0.92857143 1.        ]
#  평균 acc: 0.9714
# cross_val_precit acc: 0.3888888888888889
# ============ KNeighborsClassifier ============
# acc: [1.         0.93333333 0.92857143 0.85714286 1.         1.
#  0.92857143 1.         0.92857143 0.85714286]
#  평균 acc: 0.9433
# cross_val_precit acc: 0.9444444444444444
# ============ LabelPropagation ============
# acc: [0.93333333 0.86666667 0.71428571 0.71428571 0.71428571 0.64285714
#  0.78571429 0.85714286 0.85714286 0.71428571]
#  평균 acc: 0.78
# cross_val_precit acc: 0.6388888888888888
# ============ LabelSpreading ============
# acc: [0.93333333 0.86666667 0.71428571 0.71428571 0.71428571 0.64285714
#  0.78571429 0.85714286 0.85714286 0.71428571]
#  평균 acc: 0.78
# cross_val_precit acc: 0.6388888888888888
# ============ LinearDiscriminantAnalysis ============
# acc: [1.         0.93333333 1.         0.92857143 1.         0.85714286
#  0.92857143 1.         1.         0.92857143]
#  평균 acc: 0.9576
# cross_val_precit acc: 0.9444444444444444
# ============ LinearSVC ============
# acc: [1.         1.         1.         0.92857143 0.92857143 0.92857143
#  0.92857143 1.         1.         0.85714286] 
#  평균 acc: 0.9571
# cross_val_precit acc: 1.0
# ============ LogisticRegression ============
# acc: [1.         1.         0.92857143 0.92857143 0.92857143 1.
#  0.92857143 1.         1.         0.92857143]
#  평균 acc: 0.9643
# cross_val_precit acc: 0.9722222222222222
# ============ LogisticRegressionCV ============
# acc: [1.         0.93333333 0.92857143 0.92857143 1.         1.
#  0.92857143 1.         1.         1.        ]
#  평균 acc: 0.9719
# cross_val_precit acc: 0.9722222222222222
# ============ MLPClassifier ============
# acc: [1.         1.         0.92857143 0.92857143 0.92857143 1.
#  0.92857143 1.         1.         0.85714286]
#  평균 acc: 0.9571
# cross_val_precit acc: 0.9722222222222222
# ============ NearestCentroid ============
# acc: [1.         0.86666667 0.85714286 1.         1.         1.
#  0.92857143 1.         0.92857143 1.        ]
#  평균 acc: 0.9581
# cross_val_precit acc: 0.9444444444444444
# ============ NuSVC ============
# acc: [1.         0.93333333 1.         1.         1.         0.92857143
#  0.92857143 1.         1.         0.92857143]
#  평균 acc: 0.9719
# cross_val_precit acc: 1.0
# ============ PassiveAggressiveClassifier ============
# acc: [1.         1.         1.         0.92857143 0.92857143 0.92857143
#  0.92857143 1.         1.         0.85714286]
#  평균 acc: 0.9571
# cross_val_precit acc: 0.9722222222222222
# ============ Perceptron ============
# acc: [1.         1.         0.92857143 0.92857143 0.92857143 1.
#  0.92857143 1.         1.         0.92857143]
#  평균 acc: 0.9643
# cross_val_precit acc: 0.9722222222222222
# ============ QuadraticDiscriminantAnalysis ============
# acc: [1.         0.8        1.         1.         1.         0.92857143
#  1.         1.         1.         0.92857143]
#  평균 acc: 0.9657
# cross_val_precit acc: 0.5277777777777778
# ============ RadiusNeighborsClassifier ============
# acc: [nan nan nan nan nan nan nan nan nan nan]
#  평균 acc: nan
# ============ RandomForestClassifier ============
# acc: [1.         1.         0.92857143 1.         1.         1.
#  0.92857143 1.         0.92857143 1.        ]
#  평균 acc: 0.9786
# cross_val_precit acc: 0.9722222222222222
# ============ RidgeClassifier ============
# acc: [1.         1.         1.         0.92857143 1.         1.
#  0.92857143 1.         1.         0.92857143]
#  평균 acc: 0.9786
# cross_val_precit acc: 0.9722222222222222
# ============ RidgeClassifierCV ============
# acc: [1.         1.         1.         0.92857143 1.         1.
#  0.92857143 1.         1.         0.92857143]
#  평균 acc: 0.9786
# cross_val_precit acc: 1.0
# ============ SGDClassifier ============
# acc: [1.         1.         1.         0.92857143 0.92857143 1.
#  0.92857143 1.         1.         0.85714286]
#  평균 acc: 0.9643
# cross_val_precit acc: 1.0
# ============ SVC ============
# acc: [1.         1.         1.         1.         1.         0.92857143
#  0.92857143 1.         1.         1.        ]
#  평균 acc: 0.9857
# cross_val_precit acc: 1.0