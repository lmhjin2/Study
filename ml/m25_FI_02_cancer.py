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
# print(x.shape, y.shape)    # (569, 30) (569,)
# print(load_breast_cancer().feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, shuffle=True, random_state= 0, stratify=y)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2 모델
models = [DecisionTreeClassifier(random_state= 0), RandomForestClassifier(random_state= 0),
          GradientBoostingClassifier(random_state= 0), XGBClassifier(random_state= 0)]

np.set_printoptions(suppress=True)

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_, end='\n\n')
        ## type(model).__name__ == 모델 이름만 뽑기
        # end = '\n\n' == print('\n') 한줄 추가
        # 상위 특성 선택
        num_features_to_keep = 25
        sorted_indices = np.argsort(model.feature_importances_)[::-1]
        selected_features = sorted_indices[:num_features_to_keep]
        # 상위컬럼 데이터로 변환
        x_train_selected = x_train[:,selected_features]
        x_test_selected = x_test[:,selected_features]
        # 재학습, 평가
        model.fit(x_train_selected, y_train)
        y_predict_selected = model.predict(x_test_selected)
        accuracy_selected = accuracy_score(y_test, y_predict_selected)
        # 프린트
        print("선택된 특성 수:", num_features_to_keep)
        print("선택된 특성으로 적용한 모델의 정확도:", accuracy_selected)
        print('\n')
    except Exception as e:
        print("에러:", e)
        continue


# DecisionTreeClassifier model.score 0.9210526315789473
# DecisionTreeClassifier : [0.         0.02842052 0.         0.         0.02526269 0.
#  0.         0.82062653 0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.04836    0.
#  0.         0.         0.         0.07733026 0.         0.        ]

# 선택된 특성 수: 25
# 선택된 특성으로 적용한 모델의 정확도: 0.8947368421052632


# RandomForestClassifier model.score 0.9276315789473685
# RandomForestClassifier : [0.02631644 0.0090052  0.05458572 0.03454084 0.00442269 0.00853531
#  0.09352623 0.12210822 0.0030574  0.00304569 0.01426653 0.00332299
#  0.00724675 0.0225937  0.0044966  0.00341057 0.00748294 0.00968729
#  0.00395499 0.00231087 0.08386371 0.00512759 0.18001399 0.06035792
#  0.00972629 0.00753468 0.05719816 0.14219753 0.00758504 0.0084781 ]

# 선택된 특성 수: 25
# 선택된 특성으로 적용한 모델의 정확도: 0.9364035087719298


# GradientBoostingClassifier model.score 0.9320175438596491
# GradientBoostingClassifier : [0.00189356 0.00011999 0.00033731 0.00111337 0.00344092 0.
#  0.00260799 0.82309278 0.00000001 0.00000397 0.00017938 0.00035362
#  0.00030141 0.00000055 0.00551943 0.00000024 0.00153275 0.
#  0.00348288 0.00135898 0.014938   0.06425103 0.02743407 0.02058313
#  0.0010052  0.00680301 0.00439784 0.01308835 0.00049405 0.00166616]

# 선택된 특성 수: 25
# 선택된 특성으로 적용한 모델의 정확도: 0.9320175438596491


# XGBClassifier model.score 0.9407894736842105
# XGBClassifier : [0.01525851 0.00410022 0.         0.         0.         0.
#  0.00940113 0.44402042 0.         0.         0.         0.
#  0.         0.00167954 0.         0.         0.         0.
#  0.         0.         0.05293297 0.0373034  0.20759669 0.02476686
#  0.01075288 0.         0.02781957 0.16436787 0.         0.        ]

# 선택된 특성 수: 25
# 선택된 특성으로 적용한 모델의 정확도: 0.9342105263157895