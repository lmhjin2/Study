# https://dacon.io/competitions/open/236070/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#1
path = "c:/_data/dacon/iris/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv #.drop(['species'], axis = 1)
y = train_csv['species']
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2
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
        num_features_to_keep = 4
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
            accuracy_selected = accuracy_score(y_test, y_predict_selected)
            print("컬럼 줄인", type(model).__name__, "모델의 정확도:", accuracy_selected)
        
        print('\n')
    except Exception as e:
        print("에러:", e)
        continue


# DecisionTreeClassifier 모델의 정확도: 1.0
# 특성 중요도: [0.         0.         0.50032563 0.         0.49967437]
# 선택된 특성 수: 4
# 컬럼 줄인 DecisionTreeClassifier 모델의 정확도: 1.0


# RandomForestClassifier 모델의 정확도: 1.0
# 특성 중요도: [0.03343493 0.00692365 0.2869652  0.24708801 0.42558821]
# 선택된 특성 수: 4
# 컬럼 줄인 RandomForestClassifier 모델의 정확도: 1.0


# GradientBoostingClassifier 모델의 정확도: 1.0
# 특성 중요도: [0.         0.         0.23087303 0.13306757 0.6360594 ]
# 선택된 특성 수: 4
# 컬럼 줄인 GradientBoostingClassifier 모델의 정확도: 1.0


# XGBClassifier 모델의 정확도: 1.0
# 특성 중요도: [0.00019785 0.00356403 0.2449381  0.         0.7513    ]
# 선택된 특성 수: 4
# 컬럼 줄인 XGBClassifier 모델의 정확도: 1.0