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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2
models = [DecisionTreeRegressor(random_state= 0), RandomForestRegressor(random_state= 0),
          GradientBoostingRegressor(random_state= 0), XGBRegressor(random_state= 0)]

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "r2 score", r2_score(y_predict, y_test))
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_, end='\n\n')
        ## type(model).__name__ == 모델 이름만 뽑기
        # end = '\n\n' == print('\n') 한줄 추가
    except Exception as e:
        print("에러:", e)
        continue

# 최적의 매개변수 :  RandomForestRegressor(n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 2}
# best_score : 0.8117560013468781
# model.score : 0.8110732528984922
# r2_score: 0.8110732528984922
# 최적 튠 R2: 0.8110732528984922
# 걸린시간: 156.97 초

# DecisionTreeRegressor r2 score 0.6174329720487597
# DecisionTreeRegressor model.score 0.6272926026169132
# DecisionTreeRegressor : [0.5218018  0.05339749 0.05275489 0.02633996 0.03248298 0.12972827
#  0.08809633 0.09539829]

# RandomForestRegressor r2 score 0.7537426848866107
# RandomForestRegressor model.score 0.8094045957505897
# RandomForestRegressor : [0.52366743 0.05262664 0.05287442 0.02778925 0.03226622 0.13032447
#  0.08954688 0.0909047 ]

# GradientBoostingRegressor r2 score 0.7131243655416415
# GradientBoostingRegressor model.score 0.7862973452347977
# GradientBoostingRegressor : [0.59661078 0.02952792 0.02894911 0.00425026 0.00330772 0.11928432
#  0.09363868 0.12443121]

# XGBRegressor r2 score 0.8211145459282574
# XGBRegressor model.score 0.8442317685995901
# XGBRegressor : [0.48233894 0.06352142 0.04919809 0.02387741 0.02384355 0.1421112
#  0.10367327 0.11143605]