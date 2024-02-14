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

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 1)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
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

# ----------
# iter: 3
# n_candidates: 3
# n_resources: 351
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# c:\Users\AIA\anaconda3\envs\tf290gpu\lib\site-packages\sklearn\model_selection\_split.py:684: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
#   warnings.warn(
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_leaf=5)
# 최적의 파라미터 :  {'max_depth': 10, 'min_samples_leaf': 5}
# best_score : 0.46630723859494444
# model.score : 0.33819940383645386
# r2_score: 0.33819940383645386
# 최적 튠 R2: 0.33819940383645386
# 걸린시간: 4.1 초

# DecisionTreeRegressor accuracy score -0.22597054366821778
# DecisionTreeRegressor model.score -0.2471473515268876
# DecisionTreeRegressor : [0.05142539 0.02613193 0.35947563 0.11750633 0.07023149 0.06085748
#  0.02810653 0.03833231 0.1937202  0.05421271]

# RandomForestRegressor accuracy score -0.45545914660015385
# RandomForestRegressor model.score 0.27454472427110677
# RandomForestRegressor : [0.05658985 0.01185235 0.31936517 0.11881664 0.04230416 0.04186389
#  0.04750551 0.02722755 0.26951248 0.0649624 ]

# GradientBoostingRegressor accuracy score -0.3092658847722032
# GradientBoostingRegressor model.score 0.28959743715858377
# GradientBoostingRegressor : [0.03238974 0.01826308 0.34699758 0.12723599 0.04246017 0.04122778
#  0.03334123 0.03933527 0.27158532 0.04716383]

# XGBRegressor accuracy score -0.5109193671160241
# XGBRegressor model.score 0.12475668653462724
# XGBRegressor : [0.02937298 0.06713533 0.26994804 0.10587011 0.05830657 0.04529607
#  0.03794642 0.11817201 0.19436061 0.07359192]