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

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, OneHotEncoder

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

# ohe = OneHotEncoder()
# y_train = ohe.fit_transform(y_train.values.reshape(-1,1)).toarray()
# y_test = ohe.transform(y_test.values.reshape(-1,1)).toarray()

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
        print(type(model).__name__, "모델의 정확도:", results)
        
        # 특성 중요도 출력
        if hasattr(model, 'feature_importances_'):
            print("특성 중요도:", model.feature_importances_)
        
        # 선택된 특성 수 출력
        num_features_to_keep = 7
        if hasattr(model, 'feature_importances_'):
            sorted_indices = np.argsort(model.feature_importances_)[::-1]
            selected_features = sorted_indices[:num_features_to_keep]
            print("선택된 특성 수:", len(selected_features))
        
            # 선택된 특성으로 다시 모델 훈련 및 평가
            x_train_selected = x_train.iloc[:, selected_features]
            x_test_selected = x_test.iloc[:, selected_features]
            model_selected = model.__class__(random_state=0)
            model_selected.fit(x_train_selected, y_train)
            y_predict_selected = model_selected.predict(x_test_selected)
            r2_selected = accuracy_score(y_test, y_predict_selected)
            print("컬럼 줄인", type(model).__name__, "모델의 정확도:", r2_selected)
        
        print('\n')
    except Exception as e:
        print("에러:", e)
        continue

# DecisionTreeClassifier 모델의 정확도: 0.5818181818181818
# 특성 중요도: [0.07530933 0.1173636  0.08266283 0.07468895 0.08382683 0.07966282
#  0.08629081 0.07472353 0.08867744 0.09096016 0.14504283 0.00079088]
# 선택된 특성 수: 7
# 컬럼 줄인 DecisionTreeClassifier 모델의 정확도: 0.5945454545454546


# RandomForestClassifier 모델의 정확도: 0.5418181818181819
# 특성 중요도: [0.07488833 0.102336   0.07884249 0.08344033 0.08581635 0.0837408
#  0.09026377 0.099635   0.08335988 0.08838756 0.12580421 0.00348528]
# 선택된 특성 수: 7
# 컬럼 줄인 RandomForestClassifier 모델의 정확도: 0.5354545454545454


# 에러: y should be a 1d array, got an array of shape (4397, 7) instead.
# XGBClassifier 모델의 정확도: 0.5372727272727272
# 특성 중요도: [0.05775603 0.08979715 0.06077373 0.066026   0.06015197 0.06579991
#  0.06395647 0.0578346  0.06158298 0.06858748 0.15537561 0.19235805]
# 선택된 특성 수: 7
# 컬럼 줄인 XGBClassifier 모델의 정확도: 0.5281818181818182

# DecisionTreeClassifier 모델의 정확도: 0.5818181818181818
# 특성 중요도: [0.07530933 0.1173636  0.08266283 0.07468895 0.08382683 0.07966282
#  0.08629081 0.07472353 0.08867744 0.09096016 0.14504283 0.00079088]
# 선택된 특성 수: 7
# 컬럼 줄인 DecisionTreeClassifier 모델의 정확도: 0.5927272727272728


# RandomForestClassifier 모델의 정확도: 0.6672727272727272
# 특성 중요도: [0.07606469 0.10098247 0.07882984 0.0838592  0.08562039 0.08410614
#  0.09007932 0.10046578 0.08327781 0.08770945 0.12556684 0.00343806]
# 선택된 특성 수: 7
# 컬럼 줄인 RandomForestClassifier 모델의 정확도: 0.6563636363636364


# GradientBoostingClassifier 모델의 정확도: 0.5609090909090909
# 특성 중요도: [0.04142458 0.08757653 0.01782042 0.03677855 0.34520754 0.08499518
#  0.03287628 0.03442134 0.05211696 0.0352554  0.22693391 0.00459332]
# 선택된 특성 수: 7
# 컬럼 줄인 GradientBoostingClassifier 모델의 정확도: 0.5609090909090909