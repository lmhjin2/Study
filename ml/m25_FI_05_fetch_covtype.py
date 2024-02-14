import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import all_estimators
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = ohe.fit_transform(y_test.reshape(-1,1)).toarray()

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


# DecisionTreeClassifier model.score 0.9386848876535029
# DecisionTreeClassifier : [0.33855947 0.02653786 0.0161518  0.06183562 0.04459977 0.15100172
#  0.02748062 0.03370249 0.02370168 0.14179714 0.00852845 0.00563705
#  0.01260533 0.00210453 0.00014004 0.01006295 0.00217945 0.01173724
#  0.00045997 0.00064446 0.         0.00012113 0.00011942 0.00290132
#  0.0018041  0.00068939 0.00304903 0.00007479 0.         0.00114032
#  0.00133756 0.         0.00100253 0.00309206 0.00055822 0.00864116
#  0.0090217  0.00501496 0.0000527  0.00040934 0.00068701 0.00011879
#  0.00731444 0.00259193 0.00416454 0.01329487 0.00483006 0.00031708
#  0.00110027 0.00002005 0.0002727  0.00219151 0.00366776 0.00093158]

# 선택된 특성 수: 10
# 컬럼 줄인 DecisionTreeClassifier 의 정확도: 0.917618305895717


# RandomForestClassifier model.score 0.9544676127122363
# RandomForestClassifier : [0.24412066 0.04784682 0.03266008 0.06064151 0.05725787 0.11767427
#  0.04076168 0.04273007 0.04140066 0.11152981 0.01104291 0.0050421
#  0.01201943 0.03390591 0.00110568 0.00897016 0.00220846 0.01199452
#  0.00050563 0.00256185 0.00001092 0.00004067 0.00014502 0.01014039
#  0.00293661 0.00995217 0.00393562 0.00036665 0.00000584 0.00077524
#  0.00169359 0.00020972 0.00096859 0.0019582  0.00073849 0.0145306
#  0.00968305 0.00395118 0.00018718 0.0004928  0.00064333 0.00020417
#  0.00540014 0.0035265  0.00372466 0.0055677  0.00467661 0.00059467
#  0.00147659 0.00007323 0.00053409 0.00975566 0.00961621 0.00550382]

# 선택된 특성 수: 10
# 컬럼 줄인 RandomForestClassifier 의 정확도: 0.9544245845632212


# GradientBoostingClassifier model.score 0.7722864297823636
# GradientBoostingClassifier : [0.64531961 0.00738636 0.0011527  0.03966314 0.00812516 0.05459913
#  0.00717    0.02640119 0.00312011 0.04202303 0.02638971 0.00667158
#  0.01327996 0.00213934 0.00026556 0.01265307 0.00465864 0.01697583
#  0.00095147 0.00148205 0.         0.         0.00003705 0.00196107
#  0.00119575 0.0060487  0.00170119 0.00031337 0.00000471 0.00042466
#  0.00143701 0.00006148 0.00023133 0.00138893 0.00097457 0.01690064
#  0.01138717 0.00127024 0.         0.00007659 0.00127078 0.00006601
#  0.00543657 0.00186211 0.00310795 0.00856464 0.00071597 0.0006686
#  0.00139188 0.00000416 0.00078036 0.00309408 0.0063291  0.00086569]

# 선택된 특성 수: 10
# 컬럼 줄인 GradientBoostingClassifier 의 정확도: 0.7559357331566311


# 에러: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3 4 5 6], got [1 2 3 4 5 6 7]


