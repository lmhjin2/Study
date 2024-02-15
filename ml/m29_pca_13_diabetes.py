import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_squared_log_error, accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression, ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators
from xgboost import XGBRegressor
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape)        # (442, 10)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 1)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
pca = PCA(n_components=8)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
#2
models = [DecisionTreeRegressor(random_state= 0), RandomForestRegressor(random_state= 0),
          GradientBoostingRegressor(random_state= 0), XGBRegressor(random_state= 0)]

np.set_printoptions(suppress=True)

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_, end='\n\n')

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
        r2_selected = r2_score(y_test, y_predict_selected)

        # 프린트
        print("컬럼 줄인", type(model).__name__,"의 정확도:", r2_selected)
        print('\n')
    except Exception as e:
        print("에러:", e)
        continue

evr = pca.explained_variance_ratio_
print(np.cumsum(evr))

# DecisionTreeRegressor model.score -0.3186559905298112
# DecisionTreeRegressor : [0.44251429 0.06910964 0.09266658 0.1981019  0.05392153 0.06194734
#  0.06102642 0.02071231]

# 선택된 특성 수: 8
# 컬럼 줄인 DecisionTreeRegressor 의 정확도: -0.36816910772590905


# RandomForestRegressor model.score 0.3481006795397781
# RandomForestRegressor : [0.39823102 0.07971045 0.10852278 0.1940426  0.05468607 0.05358133
#  0.06559992 0.04562582]

# 선택된 특성 수: 8
# 컬럼 줄인 RandomForestRegressor 의 정확도: 0.3477970462470079


# GradientBoostingRegressor model.score 0.33577092693859767
# GradientBoostingRegressor : [0.43720087 0.076685   0.10214793 0.22114333 0.04638791 0.04908494
#  0.04053465 0.02681537]

# 선택된 특성 수: 8
# 컬럼 줄인 GradientBoostingRegressor 의 정확도: 0.32703122872907464


# XGBRegressor model.score 0.2442660861800816
# XGBRegressor : [0.2820865  0.08338049 0.10345201 0.21974143 0.08481009 0.08056478
#  0.1070846  0.03887999]

# 선택된 특성 수: 8
# 컬럼 줄인 XGBRegressor 의 정확도: 0.24075130695280111