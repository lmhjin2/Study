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
from sklearn.decomposition import PCA
#1
path = "c:/_data/dacon/iris/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv #.drop(['species'], axis = 1)
y = train_csv['species']
# print(x.shape)        # (120,5)
scaler = StandardScaler()
x = scaler.fit_transform(x)
pca = PCA(n_components=4)
x = pca.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)


#2
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
        num_features_to_keep = 40
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


# DecisionTreeClassifier model.score 1.0
# DecisionTreeClassifier : [0.96965924 0.         0.         0.03034076]

# 선택된 특성 수: 4
# 컬럼 줄인 DecisionTreeClassifier 의 정확도: 1.0


# RandomForestClassifier model.score 1.0
# RandomForestClassifier : [0.66173468 0.07524558 0.06674317 0.19627657]

# 선택된 특성 수: 4
# 컬럼 줄인 RandomForestClassifier 의 정확도: 1.0


# GradientBoostingClassifier model.score 1.0
# GradientBoostingClassifier : [0.75704397 0.07239965 0.01739605 0.15316033]

# 선택된 특성 수: 4
# 컬럼 줄인 GradientBoostingClassifier 의 정확도: 1.0


# XGBClassifier model.score 1.0
# XGBClassifier : [0.7102415  0.11683508 0.01261061 0.16031285]

# 선택된 특성 수: 4
# 컬럼 줄인 XGBClassifier 의 정확도: 1.0