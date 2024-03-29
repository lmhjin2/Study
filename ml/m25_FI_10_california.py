import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,mean_absolute_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

plt.rcParams['font.family'],"Malgun Gothic"
plt.rcParams['axes.unicode_minus']=False

#1 
datasets = fetch_california_housing()
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
        num_features_to_keep = 6
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

# DecisionTreeRegressor model.score 0.6272926026169132
# DecisionTreeRegressor : [0.5218018  0.05339749 0.05275489 0.02633996 0.03248298 0.12972827
#  0.08809633 0.09539829]

# 선택된 특성 수: 6
# 컬럼 줄인 DecisionTreeRegressor 의 정확도: 0.6198107538238058


# RandomForestRegressor model.score 0.8094045957505897
# RandomForestRegressor : [0.52366743 0.05262664 0.05287442 0.02778925 0.03226622 0.13032447
#  0.08954688 0.0909047 ]

# 선택된 특성 수: 6
# 컬럼 줄인 RandomForestRegressor 의 정확도: 0.8152804499656907


# GradientBoostingRegressor model.score 0.7862973452347977
# GradientBoostingRegressor : [0.59661078 0.02952792 0.02894911 0.00425026 0.00330772 0.11928432
#  0.09363868 0.12443121]

# 선택된 특성 수: 6
# 컬럼 줄인 GradientBoostingRegressor 의 정확도: 0.7850895823036456


# XGBRegressor model.score 0.8442317685995901
# XGBRegressor : [0.48233894 0.06352142 0.04919809 0.02387741 0.02384355 0.1421112
#  0.10367327 0.11143605]

# 선택된 특성 수: 6
# 컬럼 줄인 XGBRegressor 의 정확도: 0.851658363966794