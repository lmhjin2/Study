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
from sklearn.decomposition import PCA
#1
datasets = load_boston()

x = datasets.data
y = datasets.target

# 라벨 인코딩. StratifiedKFold 할때만 필요
label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)
# 라벨 인코딩. StratifiedKFold 할때만 필요
# print(x.shape, y.shape)   # (506, 13)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
pca = PCA(n_components=12)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
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
        num_features_to_keep = 12
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

# DecisionTreeRegressor model.score 0.5937896723433742
# DecisionTreeRegressor : [0.55910411 0.0019135  0.25016708 0.00617164 0.01605237 0.00792909
#  0.02348934 0.02590795 0.02700738 0.02190116 0.02651937 0.033837  ]

# 선택된 특성 수: 12
# 컬럼 줄인 DecisionTreeRegressor 의 정확도: 0.4600104057766897


# RandomForestRegressor model.score 0.6535641790665856
# RandomForestRegressor : [0.56457132 0.00770739 0.24099562 0.01090731 0.02609572 0.00995914
#  0.0286064  0.02856513 0.01521272 0.01774286 0.01627539 0.03336101]

# 선택된 특성 수: 12
# 컬럼 줄인 RandomForestRegressor 의 정확도: 0.6510374543564214


# GradientBoostingRegressor model.score 0.6273038677638332
# GradientBoostingRegressor : [0.56657473 0.00925328 0.24751365 0.01015738 0.02674875 0.00927242
#  0.01948786 0.03306341 0.01874475 0.02380236 0.01217321 0.02320818]

# 선택된 특성 수: 12
# 컬럼 줄인 GradientBoostingRegressor 의 정확도: 0.6244718725282874


# XGBRegressor model.score 0.647608996861649
# XGBRegressor : [0.3980177  0.00782529 0.29362336 0.00726009 0.04289748 0.01165963
#  0.03233406 0.04629359 0.04644198 0.02911012 0.02578815 0.0587485 ]

# 선택된 특성 수: 12
# 컬럼 줄인 XGBRegressor 의 정확도: 0.6626247172944315