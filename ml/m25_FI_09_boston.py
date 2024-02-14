import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#1
datasets = load_boston()

x = datasets.data
y = datasets.target

# 라벨 인코딩. StratifiedKFold 할때만 필요
label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)
# 라벨 인코딩. StratifiedKFold 할때만 필요
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2 모델
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
        num_features_to_keep = 10
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


# DecisionTreeRegressor model.score 0.6165248747431489
# DecisionTreeRegressor : [0.02803913 0.0000053  0.01735001 0.00143646 0.04138179 0.15881344
#  0.0225367  0.02109152 0.00224402 0.01219388 0.01077829 0.0047105
#  0.67941896]

# 선택된 특성 수: 10
# 컬럼 줄인 DecisionTreeRegressor 의 정확도: 0.6147180604941925


# RandomForestRegressor model.score 0.7418415344287195
# RandomForestRegressor : [0.05243644 0.001352   0.00863553 0.00085234 0.02121723 0.1763693
#  0.01447535 0.02985291 0.00249164 0.01548248 0.0182484  0.01329596
#  0.64529042]

# 선택된 특성 수: 10
# 컬럼 줄인 RandomForestRegressor 의 정확도: 0.740213696219874


# GradientBoostingRegressor model.score 0.7362122111949143
# GradientBoostingRegressor : [0.02794375 0.00015314 0.00317765 0.00040254 0.02668757 0.15962443
#  0.00923672 0.03536407 0.0003996  0.01278681 0.04771356 0.01052942
#  0.66598074]

# 선택된 특성 수: 10
# 컬럼 줄인 GradientBoostingRegressor 의 정확도: 0.7202414164308785


# XGBRegressor model.score 0.7622746562973528
# XGBRegressor : [0.02382473 0.00130383 0.02772509 0.01033411 0.05156561 0.10908394
#  0.01620274 0.0178323  0.0057355  0.03537114 0.02386856 0.00930779
#  0.66784465]

# 선택된 특성 수: 10
# 컬럼 줄인 XGBRegressor 의 정확도: 0.7506577472287113