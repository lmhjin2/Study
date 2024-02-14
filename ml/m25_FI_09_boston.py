# restore_best_weights 와
# save_best_only 에 대한 고찰

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2 모델
models = [DecisionTreeRegressor(random_state= 0), RandomForestRegressor(random_state= 0),
          GradientBoostingRegressor(random_state= 0), XGBRegressor(random_state= 0)]

# np.set_printoptions(suppress=True)

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

# Pipeline
# model.score : 0.7524490468170687
# r2_score: 0.7524490468170687
# 걸린시간: 0.07 초

# DecisionTreeRegressor r2 score 0.6254936466858083
# DecisionTreeRegressor model.score 0.6165248747431489
# DecisionTreeRegressor : [2.80391263e-02 5.30081536e-06 1.73500104e-02 1.43646413e-03
#  4.13817924e-02 1.58813444e-01 2.25367034e-02 2.10915163e-02
#  2.24402025e-03 1.21938754e-02 1.07782890e-02 4.71049759e-03
#  6.79418960e-01]

# RandomForestRegressor r2 score 0.658086048366294
# RandomForestRegressor model.score 0.7418415344287195
# RandomForestRegressor : [0.05243644 0.001352   0.00863553 0.00085234 0.02121723 0.1763693
#  0.01447535 0.02985291 0.00249164 0.01548248 0.0182484  0.01329596
#  0.64529042]

# GradientBoostingRegressor r2 score 0.6527391158359996
# GradientBoostingRegressor model.score 0.7362122111949143
# GradientBoostingRegressor : [2.79437518e-02 1.53135969e-04 3.17765328e-03 4.02544684e-04
#  2.66875679e-02 1.59624425e-01 9.23671737e-03 3.53640662e-02
#  3.99603768e-04 1.27868130e-02 4.77135577e-02 1.05294229e-02
#  6.65980740e-01]

# XGBRegressor r2 score 0.6879833206583839
# XGBRegressor model.score 0.7622746562973528
# XGBRegressor : [0.02382473 0.00130383 0.02772509 0.01033411 0.05156561 0.10908394
#  0.01620274 0.0178323  0.0057355  0.03537114 0.02386856 0.00930779
#  0.66784465]