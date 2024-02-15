import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#1

#1
datasets = load_wine()
x = datasets.data
y = datasets.target

x = x.astype(np.float32)
y = y.astype(np.float32)
# print(x.shape, y.shape)   # (178, 13)

scaler = StandardScaler()
x = scaler.fit_transform(x)
pca = PCA(n_components=10)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

models = [DecisionTreeClassifier(random_state= 0), RandomForestClassifier(random_state= 0),
          GradientBoostingClassifier(random_state= 0), XGBClassifier(random_state= 0)]

np.set_printoptions(suppress=True)

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_)

        # 남길 상위 특성 선택
        num_features_to_keep = 8
        sorted_indices = np.argsort(model.feature_importances_)[::-1]
        selected_features = sorted_indices[:num_features_to_keep]

        # 선택된 특성 수 출력
        print("선택된 특성 수:", len(selected_features))

        # 상위컬럼 데이터로 변환
        x_train_selected = x_train[:, selected_features]
        x_test_selected = x_test[:, selected_features]

        # 재학습, 평가
        model_selected = model.__class__(random_state=0)
        model_selected.fit(x_train_selected, y_train)
        y_predict_selected = model_selected.predict(x_test_selected)
        accuracy_selected = accuracy_score(y_test, y_predict_selected)

        # 프린트
        print("컬럼 줄인", type(model).__name__,"의 정확도:", accuracy_selected)
        print('\n')
    except Exception as e:
        print("에러:", e)
        continue

# DecisionTreeClassifier model.score 0.9722222222222222
# DecisionTreeClassifier : [0.44014694 0.46155997 0.         0.         0.07320809 0.
#  0.         0.025085   0.         0.        ]
# 선택된 특성 수: 8
# 컬럼 줄인 DecisionTreeClassifier 의 정확도: 0.9444444444444444


# RandomForestClassifier model.score 0.9444444444444444
# RandomForestClassifier : [0.34417887 0.38753581 0.03770863 0.02623851 0.04230586 0.03899558
#  0.03790202 0.03099527 0.03472095 0.0194185 ]
# 선택된 특성 수: 8
# 컬럼 줄인 RandomForestClassifier 의 정확도: 0.9722222222222222


# GradientBoostingClassifier model.score 0.9722222222222222
# GradientBoostingClassifier : [0.47360181 0.40229697 0.02625738 0.01137791 0.03416737 0.02556084
#  0.00162363 0.00756435 0.01423837 0.00331137]
# 선택된 특성 수: 8
# 컬럼 줄인 GradientBoostingClassifier 의 정확도: 0.9722222222222222


# XGBClassifier model.score 0.9722222222222222
# XGBClassifier : [0.43152648 0.27614516 0.0529668  0.05704446 0.03326662 0.04037267
#  0.01660395 0.03777455 0.02104259 0.03325676]
# 선택된 특성 수: 8
# 컬럼 줄인 XGBClassifier 의 정확도: 0.9722222222222222