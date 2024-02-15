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
scaler = StandardScaler()
x = scaler.fit_transform(x)
pca = PCA(n_components=28)
x = pca.fit_transform(x)

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

# DecisionTreeClassifier model.score 0.8991228070175439
# DecisionTreeClassifier : [0.81877303 0.04639485 0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.01894702
#  0.         0.         0.07221675 0.         0.         0.
#  0.01840567 0.         0.02526269 0.        ]

# 선택된 특성 수: 25
# 선택된 특성으로 적용한 모델의 정확도: 0.881578947368421


# RandomForestClassifier model.score 0.918859649122807
# RandomForestClassifier : [0.30987412 0.08020279 0.07262663 0.00782145 0.01322984 0.0134716
#  0.05119075 0.01356333 0.0134418  0.01227977 0.03106397 0.02165573
#  0.03454934 0.00674477 0.01801372 0.01374741 0.02194863 0.01815259
#  0.01036292 0.01362188 0.03832686 0.01495143 0.02167091 0.03989311
#  0.04200611 0.02220444 0.01075583 0.03262825]

# 선택된 특성 수: 25
# 선택된 특성으로 적용한 모델의 정확도: 0.9144736842105263


# GradientBoostingClassifier model.score 0.8859649122807017
# GradientBoostingClassifier : [0.81771092 0.01887502 0.00111972 0.0053454  0.02739932 0.00283106
#  0.0153917  0.00140015 0.00190261 0.03745104 0.00049612 0.00007719
#  0.         0.         0.         0.00181417 0.00134492 0.00043262
#  0.00075723 0.00000134 0.05234799 0.00201358 0.00569725 0.000216
#  0.00446352 0.         0.00002158 0.00088956]

# 선택된 특성 수: 25
# 선택된 특성으로 적용한 모델의 정확도: 0.8969298245614035


# XGBClassifier model.score 0.9342105263157895
# XGBClassifier : [0.48601228 0.07898185 0.01845046 0.0101773  0.04783645 0.
#  0.04030128 0.         0.         0.         0.         0.017585
#  0.0007151  0.         0.06754123 0.07722399 0.         0.
#  0.         0.02612287 0.0822205  0.00907286 0.00652846 0.
#  0.02642781 0.00344542 0.00135709 0.        ]

# 선택된 특성 수: 25
# 선택된 특성으로 적용한 모델의 정확도: 0.9342105263157895