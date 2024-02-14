import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators
from xgboost import XGBClassifier
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, shuffle=True, random_state= 0, stratify=y)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2 모델
models = [DecisionTreeClassifier(random_state= 0), RandomForestClassifier(random_state= 0),
          GradientBoostingClassifier(random_state= 0), XGBClassifier(random_state= 0)]

# np.set_printoptions(suppress=True)

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "accuracy score", accuracy_score(y_predict, y_test))
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_, end='\n\n')
        ## type(model).__name__ == 모델 이름만 뽑기
        # end = '\n\n' == print('\n') 한줄 추가
    except Exception as e:
        print("에러:", e)
        continue

# np.set_printoptions(suppress=False)

# BaggingClassifier 의 정답률: 0.9651162790697675

# DecisionTreeClassifier accuracy score 0.9210526315789473
# DecisionTreeClassifier model.score 0.9210526315789473
# DecisionTreeClassifier : [0.         0.02842052 0.         0.         0.02526269 0.
#  0.         0.82062653 0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.04836    0.
#  0.         0.         0.         0.07733026 0.         0.        ]

# RandomForestClassifier accuracy score 0.9276315789473685
# RandomForestClassifier model.score 0.9276315789473685
# RandomForestClassifier : [0.02631644 0.0090052  0.05458572 0.03454084 0.00442269 0.00853531
#  0.09352623 0.12210822 0.0030574  0.00304569 0.01426653 0.00332299
#  0.00724675 0.0225937  0.0044966  0.00341057 0.00748294 0.00968729
#  0.00395499 0.00231087 0.08386371 0.00512759 0.18001399 0.06035792
#  0.00972629 0.00753468 0.05719816 0.14219753 0.00758504 0.0084781 ]

# GradientBoostingClassifier accuracy score 0.9320175438596491
# GradientBoostingClassifier model.score 0.9320175438596491
# GradientBoostingClassifier : [1.89356476e-03 1.19988026e-04 3.37313531e-04 1.11337366e-03
#  3.44091805e-03 0.00000000e+00 2.60799404e-03 8.23092783e-01
#  1.21128357e-08 3.97417702e-06 1.79383088e-04 3.53620921e-04
#  3.01405931e-04 5.53435144e-07 5.51942774e-03 2.41392795e-07
#  1.53275432e-03 0.00000000e+00 3.48288369e-03 1.35898216e-03
#  1.49379989e-02 6.42510328e-02 2.74340721e-02 2.05831255e-02
#  1.00519604e-03 6.80300749e-03 4.39784023e-03 1.30883475e-02
#  4.94049910e-04 1.66615525e-03]

# XGBClassifier accuracy score 0.9407894736842105
# XGBClassifier model.score 0.9407894736842105
# XGBClassifier : [0.01525851 0.00410022 0.         0.         0.         0.
#  0.00940113 0.44402042 0.         0.         0.         0.
#  0.         0.00167954 0.         0.         0.         0.
#  0.         0.         0.05293297 0.0373034  0.20759669 0.02476686
#  0.01075288 0.         0.02781957 0.16436787 0.         0.        ]