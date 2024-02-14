# https://dacon.io/competitions/official/236214/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

path = "c:/_data/dacon/dechul/"
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')


le_work_period = LabelEncoder() 
le_work_period.fit(train_csv['근로기간'])
train_csv['근로기간'] = le_work_period.transform(train_csv['근로기간'])
test_csv['근로기간'] = le_work_period.transform(test_csv['근로기간'])

le_grade = LabelEncoder()
le_grade.fit(train_csv['대출등급'])
train_csv['대출등급'] = le_grade.transform(train_csv['대출등급'])

le_purpose = LabelEncoder()
test_csv.iloc[34486,7] = '이사'     # 결혼 -> 이사 로 임의로 바꿈
le_purpose.fit(train_csv['대출목적'])
train_csv['대출목적'] = le_purpose.transform(train_csv['대출목적'])
test_csv['대출목적'] = le_purpose.transform(test_csv['대출목적'])

le_own = LabelEncoder()
le_own.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = le_own.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = le_own.transform(test_csv['주택소유상태'])

le_loan_period = LabelEncoder()
le_loan_period.fit(train_csv['대출기간'])
train_csv['대출기간'] = le_loan_period.transform(train_csv['대출기간'])
test_csv['대출기간'] = le_loan_period.transform(test_csv['대출기간'])

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size = 0.18, random_state = 1818 )
# 1785 / 1818 / 

from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = RobustScaler()

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

#3
# import datetime

# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")   # 월일_시분

# path1 = "c:/_data/_save/MCP/k28/11/"
# filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
# filepath = "".join([path1, 'k28_', date, '_1_', filename])


#4
# y_submit = model.predict(test_csv)

# y_submit = le_grade.inverse_transform(y_submit)

# submission_csv['대출등급'] = y_submit
# submission_csv.to_csv(path + "submission_0208_1.csv", index=False)
# https://dacon.io/competitions/official/236214/mysubmission


# pipeline
# model.score : 0.8077078405353949
# accuracy_score: 0.8077078405353949
# 걸린시간: 7.98 초

# DecisionTreeClassifier accuracy score 0.8330352506778976
# DecisionTreeClassifier model.score 0.8330352506778976
# DecisionTreeClassifier : [6.06270412e-02 3.43817418e-02 1.58744273e-02 5.67929282e-03
#  3.43257650e-02 3.11176293e-02 2.41235605e-02 7.93268303e-03
#  5.96468227e-03 4.14263961e-01 3.64971715e-01 4.26936091e-04
#  3.10564802e-04]

# RandomForestClassifier accuracy score 0.8032077539952691
# RandomForestClassifier model.score 0.8032077539952691
# RandomForestClassifier : [0.09944017 0.02874133 0.04832982 0.0173361  0.08326313 0.08995178
#  0.07126249 0.02577123 0.0164481  0.26129073 0.25668961 0.00056213
#  0.00091339]

# GradientBoostingClassifier accuracy score 0.7451681763110829
# GradientBoostingClassifier model.score 0.7451681763110829
# GradientBoostingClassifier : [1.97606253e-02 1.15451092e-01 1.38571497e-05 8.15199290e-04
#  1.57970654e-02 6.23591904e-03 1.37801159e-03 6.26585870e-03
#  1.15958410e-03 4.02163934e-01 4.30906674e-01 5.21796633e-05
#  0.00000000e+00]

# XGBClassifier accuracy score 0.8509779034212196
# XGBClassifier model.score 0.8509779034212196
# XGBClassifier : [0.04340876 0.42047614 0.0114096  0.01663587 0.03177512 0.01676813
#  0.01319469 0.02497237 0.01959267 0.1859271  0.19321862 0.01020237
#  0.01241865]