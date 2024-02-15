# https://dacon.io/competitions/open/235610/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

path = "c:/_data/dacon/wine/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")


train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x.shape)    # (5497, 12)
pca = PCA(n_components=10)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)
# 끄면 xgboost에러 키면 GradientBoost 에러
ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.values.reshape(-1,1)).toarray()
y_test = ohe.transform(y_test.values.reshape(-1,1)).toarray()

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

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

        # 남길 상위 특성 선택
        num_features_to_keep = 7
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

# DecisionTreeClassifier model.score 0.5818181818181818
# DecisionTreeClassifier : [0.07902069 0.12360981 0.11741524 0.09917256 0.10259327 0.09701177
#  0.07545029 0.08596673 0.1271675  0.09259215]

# 선택된 특성 수: 7
# 컬럼 줄인 DecisionTreeClassifier 의 정확도: 0.5781818181818181


# RandomForestClassifier model.score 0.6581818181818182
# RandomForestClassifier : [0.09390673 0.13045735 0.10651162 0.10046615 0.10057886 0.09815008
#  0.08990982 0.08774995 0.10827393 0.08399551]

# 선택된 특성 수: 7
# 컬럼 줄인 RandomForestClassifier 의 정확도: 0.6436363636363637


# GradientBoostingClassifier model.score 0.5890909090909091
# GradientBoostingClassifier : [0.08373995 0.21363015 0.11050703 0.09758415 0.10478605 0.0880156
#  0.0563206  0.06581436 0.13214673 0.04745538]

# 선택된 특성 수: 7
# 컬럼 줄인 GradientBoostingClassifier 의 정확도: 0.5563636363636364

# XGBClassifier model.score 0.5372727272727272
# XGBClassifier : [0.10111936 0.13528964 0.1029093  0.09687714 0.10211872 0.09238026
#  0.08678181 0.09054778 0.10613011 0.08584585]

# 선택된 특성 수: 7
# 컬럼 줄인 XGBClassifier 의 정확도: 0.5136363636363637