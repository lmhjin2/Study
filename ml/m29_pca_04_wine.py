import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#1
import numpy as np
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

#1
datasets = load_wine()
x = datasets.data
y = datasets.target

x = x.astype(np.float32)
y = y.astype(np.float32)
print(x.shape, y.shape)   # (178, 13)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

models = [DecisionTreeClassifier(random_state= 0), RandomForestClassifier(random_state= 0),
          GradientBoostingClassifier(random_state= 0), XGBClassifier(random_state= 0)]

np.set_printoptions(suppress=True)

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_)

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
        accuracy_selected = accuracy_score(y_test, y_predict_selected)

        # 프린트
        print("컬럼 줄인", type(model).__name__,"의 정확도:", accuracy_selected)
        print('\n')
    except Exception as e:
        print("에러:", e)
        continue

# DecisionTreeClassifier model.score 0.9444444444444444
# DecisionTreeClassifier : [0.         0.         0.02018719 0.0209047  0.         0.
#  0.1281229  0.         0.         0.33554694 0.         0.04122818
#  0.4540101 ]

# 선택된 특성 수: 10
# 컬럼 줄인 DecisionTreeClassifier 의 정확도: 0.9722222222222222


# RandomForestClassifier model.score 1.0
# RandomForestClassifier : [0.12930939 0.02858818 0.01005337 0.02107876 0.02670528 0.04826478
#  0.17041694 0.01237242 0.01801449 0.17606478 0.06869346 0.09987653
#  0.19056163]

# 선택된 특성 수: 10
# 컬럼 줄인 RandomForestClassifier 의 정확도: 1.0


# GradientBoostingClassifier model.score 0.9722222222222222
# GradientBoostingClassifier : [0.01412274 0.01043494 0.0351641  0.00260472 0.01326166 0.00192114
#  0.26411514 0.0004652  0.00255949 0.34061822 0.0041362  0.00731209
#  0.30328436]

# 선택된 특성 수: 10
# 컬럼 줄인 GradientBoostingClassifier 의 정확도: 0.9722222222222222


# XGBClassifier model.score 0.9722222222222222
# XGBClassifier : [0.02263726 0.05461496 0.05084845 0.0081804  0.02534876 0.01520464
#  0.27298525 0.         0.001672   0.2607776  0.04897151 0.00864236
#  0.23011674]

# 선택된 특성 수: 10
# 컬럼 줄인 XGBClassifier 의 정확도: 0.9722222222222222