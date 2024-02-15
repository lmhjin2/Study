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
from sklearn.decomposition import  PCA

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
# print(x.shape, y.shape)   # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
pca = PCA(n_components=7)
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

# DecisionTreeRegressor model.score 0.4517260341923206
# DecisionTreeRegressor : [0.07070645 0.05098902 0.04743661 0.5176248  0.06680789 0.10378259
#  0.14265263]

# 선택된 특성 수: 7
# 컬럼 줄인 DecisionTreeRegressor 의 정확도: 0.4514392677508926


# RandomForestRegressor model.score 0.7345613473710503
# RandomForestRegressor : [0.06828464 0.05187796 0.04654574 0.52048253 0.06448964 0.10000733
#  0.14831215]

# 선택된 특성 수: 7
# 컬럼 줄인 RandomForestRegressor 의 정확도: 0.7342640108643828


# GradientBoostingRegressor model.score 0.69454186085859
# GradientBoostingRegressor : [0.02386883 0.02646107 0.00368475 0.66628036 0.02582674 0.09425217
#  0.15962608]

# 선택된 특성 수: 7
# 컬럼 줄인 GradientBoostingRegressor 의 정확도: 0.69454186085859


# XGBRegressor model.score 0.735695874046558
# XGBRegressor : [0.04885179 0.04130459 0.03545461 0.58793354 0.0550629  0.09798978
#  0.13340269]

# 선택된 특성 수: 7
# 컬럼 줄인 XGBRegressor 의 정확도: 0.7356383802720532