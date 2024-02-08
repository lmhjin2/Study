import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, shuffle=True, random_state= 0, stratify=y)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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

# BaggingClassifier 의 정답률: 0.9651162790697675

# acc: [0.96491228 0.94736842 0.98245614 0.96491228 0.94736842 0.92982456
#  0.96491228 0.94736842 0.96491228 0.89285714]
#  평균 acc: 0.9507

# acc: [0.92982456 0.92982456 0.9122807  0.98245614 0.94736842 0.96491228
#  0.92982456 0.96491228 0.94736842 0.98214286]
#  평균 acc: 0.9491

# acc: [0.91666667 0.91666667 0.91666667 0.81818182 1.         0.90909091
#  0.90909091 1.         1.         0.90909091]
#  평균 acc: 0.9295
# cross_val_precit acc 0.9276315789473685

# ============ AdaBoostClassifier ============
# acc: [1.         0.91666667 0.91666667 1.         1.         1.
#  0.90909091 1.         1.         1.        ]
#  평균 acc: 0.9742
# cross_val_precit acc: 0.956140350877193
# ============ BaggingClassifier ============
# acc: [1.         0.91666667 0.91666667 0.81818182 0.90909091 0.90909091
#  1.         1.         1.         0.90909091]
#  평균 acc: 0.9379
# cross_val_precit acc: 0.9407894736842105
# ============ BernoulliNB ============
# acc: [0.66666667 0.58333333 0.58333333 0.63636364 0.63636364 0.63636364
#  0.63636364 0.63636364 0.63636364 0.63636364]
#  평균 acc: 0.6288
# cross_val_precit acc: 0.6271929824561403
# ============ CalibratedClassifierCV ============
# acc: [1.         0.83333333 0.91666667 0.90909091 0.90909091 0.90909091
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9477
# cross_val_precit acc: 0.9671052631578947
# ============ CategoricalNB ============
# acc: [0.66666667 0.58333333        nan        nan        nan        nan
#         nan        nan        nan        nan]
#  평균 acc: nan
# ============ ComplementNB ============
# acc: [1.         0.83333333 0.91666667 0.81818182 1.         0.81818182
#  1.         1.         1.         0.90909091]
#  평균 acc: 0.9295
# cross_val_precit acc: 0.9013157894736842
# ============ DecisionTreeClassifier ============
# acc: [1.         0.83333333 0.91666667 0.81818182 1.         0.90909091
#  0.90909091 1.         1.         0.90909091]
#  평균 acc: 0.9295
# cross_val_precit acc: 0.9166666666666666
# ============ DummyClassifier ============
# acc: [0.66666667 0.58333333 0.58333333 0.63636364 0.63636364 0.63636364
#  0.63636364 0.63636364 0.63636364 0.63636364]
#  평균 acc: 0.6288
# cross_val_precit acc: 0.6271929824561403
# ============ ExtraTreeClassifier ============
# acc: [0.83333333 0.83333333 0.83333333 0.72727273 1.         1.
#  1.         0.90909091 0.90909091 0.72727273]
#  평균 acc: 0.8773
# cross_val_precit acc: 0.9144736842105263
# ============ ExtraTreesClassifier ============
# acc: [1.         0.91666667 0.91666667 0.90909091 1.         0.90909091
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9652
# cross_val_precit acc: 0.9605263157894737
# ============ GaussianNB ============
# acc: [1.         0.83333333 0.91666667 0.90909091 1.         0.90909091
#  0.90909091 1.         1.         0.90909091]
#  평균 acc: 0.9386
# cross_val_precit acc: 0.9385964912280702
# ============ GaussianProcessClassifier ============
# acc: [1.         0.83333333 0.91666667 0.90909091 1.         1.
#  1.         1.         1.         0.90909091]
#  평균 acc: 0.9568
# cross_val_precit acc: 0.9517543859649122
# ============ GradientBoostingClassifier ============
# acc: [1.         0.91666667 0.91666667 0.81818182 0.90909091 0.81818182
#  0.90909091 1.         1.         0.90909091]
#  평균 acc: 0.9197
# cross_val_precit acc: 0.9605263157894737
# ============ HistGradientBoostingClassifier ============
# acc: [1.         0.91666667 0.91666667 1.         1.         1.
#  1.         1.         1.         0.90909091]
#  평균 acc: 0.9742
# cross_val_precit acc: 0.9627192982456141
# ============ KNeighborsClassifier ============
# acc: [1.         0.91666667 0.91666667 0.90909091 1.         1.
#  1.         1.         1.         0.90909091]
#  평균 acc: 0.9652
# cross_val_precit acc: 0.9583333333333334
# ============ LabelPropagation ============
# acc: [1.         0.91666667 0.91666667 0.90909091 1.         1.
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9742
# cross_val_precit acc: 0.9605263157894737
# ============ LabelSpreading ============
# acc: [1.         0.91666667 0.91666667 0.90909091 1.         1.
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9742
# cross_val_precit acc: 0.9583333333333334
# ============ LinearDiscriminantAnalysis ============
# acc: [1.         1.         0.91666667 0.90909091 0.81818182 0.90909091
#  1.         0.90909091 1.         1.        ]
#  평균 acc: 0.9462
# cross_val_precit acc: 0.9539473684210527
# ============ LinearSVC ============
# acc: [1.         0.83333333 0.91666667 0.90909091 1.         0.90909091
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9568
# cross_val_precit acc: 0.9692982456140351
# ============ LogisticRegression ============
# acc: [1.         0.83333333 0.91666667 0.90909091 1.         1.
#  1.         1.         1.         0.90909091]
#  평균 acc: 0.9568
# cross_val_precit acc: 0.9583333333333334
# ============ LogisticRegressionCV ============
# acc: [1.         0.91666667 0.91666667 0.90909091 1.         1.
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9742
# cross_val_precit acc: 0.9692982456140351
# ============ MLPClassifier ============
# acc: [1.         0.83333333 0.91666667 0.90909091 0.90909091 0.81818182
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9386
# cross_val_precit acc: 0.9583333333333334
# ============ MultinomialNB ============
# acc: [1.         0.83333333 0.83333333 0.81818182 0.90909091 0.72727273
#  1.         0.90909091 0.90909091 0.81818182]
#  평균 acc: 0.8758
# cross_val_precit acc: 0.8640350877192983
# ============ NearestCentroid ============
# acc: [1.         0.83333333 0.91666667 0.90909091 1.         1.
#  1.         1.         1.         0.90909091]
#  평균 acc: 0.9568
# cross_val_precit acc: 0.9166666666666666
# ============ NuSVC ============
# acc: [1.         0.83333333 0.91666667 0.90909091 1.         1.
#  1.         1.         1.         0.81818182]
#  평균 acc: 0.9477
# cross_val_precit acc: 0.9320175438596491
# ============ PassiveAggressiveClassifier ============
# acc: [1.         0.91666667 0.83333333 0.90909091 1.         0.90909091
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9568
# cross_val_precit acc: 0.9429824561403509
# ============ Perceptron ============
# acc: [0.66666667 0.91666667 0.91666667 1.         1.         0.90909091
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9409
# cross_val_precit acc: 0.9320175438596491
# ============ QuadraticDiscriminantAnalysis ============
# acc: [1.         0.91666667 0.91666667 1.         0.90909091 0.81818182
#  0.90909091 0.90909091 1.         0.90909091]
#  평균 acc: 0.9288
# cross_val_precit acc: 0.9517543859649122
# ============ RadiusNeighborsClassifier ============
# acc: [0.91666667 0.83333333        nan 0.81818182        nan        nan
#  1.         1.         0.90909091        nan]
#  평균 acc: nan
# ============ RandomForestClassifier ============
# acc: [1.         0.91666667 0.91666667 1.         1.         0.90909091
#  1.         1.         1.         0.90909091]
#  평균 acc: 0.9652
# cross_val_precit acc: 0.956140350877193
# ============ RidgeClassifier ============
# acc: [1.         0.91666667 0.91666667 0.90909091 0.90909091 0.90909091
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9561
# cross_val_precit acc: 0.9539473684210527
# ============ RidgeClassifierCV ============
# acc: [1.         0.91666667 0.91666667 0.90909091 0.90909091 0.90909091
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9561
# cross_val_precit acc: 0.9473684210526315
# ============ SGDClassifier ============
# acc: [1.         0.83333333 0.91666667 0.90909091 0.90909091 0.81818182
#  1.         1.         1.         0.90909091]
#  평균 acc: 0.9295
# cross_val_precit acc: 0.9298245614035088
# ============ SVC ============
# acc: [1.         0.83333333 0.91666667 0.90909091 1.         1.
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9659
# cross_val_precit acc: 0.9692982456140351