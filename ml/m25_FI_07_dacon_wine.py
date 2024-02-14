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

ohe = OneHotEncoder()
y_ohe = ohe.fit_transform(y.values.reshape(5497,1)).toarray()
print(y_ohe.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

#2 모델
models = [DecisionTreeClassifier(random_state= 0), RandomForestClassifier(random_state= 0),
          GradientBoostingClassifier(random_state= 0), XGBClassifier(random_state= 0)]

# np.set_printoptions(suppress=True)

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "accuracy score", accuracy_score(y_predict, y_test))
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_, end='\n\n')
        ## type(model).__name__ == 모델 이름만 뽑기
        # end = '\n\n' == print('\n') 한줄 추가
    except Exception as e:
        print("에러:", e)
        continue

# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3)
# 최적의 파라미터 :  {'min_samples_split': 3}
# best_score : 0.6602197745371806
# model.score : 0.6790909090909091
# accuracy_score: 0.6790909090909091
# 최적 튠 ACC: 0.6790909090909091
# 걸린시간: 162.35 초 


# DecisionTreeClassifier accuracy score 0.5818181818181818
# DecisionTreeClassifier model.score 0.5818181818181818
# DecisionTreeClassifier : [0.07530933 0.1173636  0.08266283 0.07468895 0.08382683 0.07966282
#  0.08629081 0.07472353 0.08867744 0.09096016 0.14504283 0.00079088]

# RandomForestClassifier accuracy score 0.5418181818181819
# RandomForestClassifier model.score 0.5418181818181819
# RandomForestClassifier : [0.07488833 0.102336   0.07884249 0.08344033 0.08581635 0.0837408
#  0.09026377 0.099635   0.08335988 0.08838756 0.12580421 0.00348528]

# 에러: y should be a 1d array, got an array of shape (4397, 7) instead.
# XGBClassifier accuracy score 0.5372727272727272
# XGBClassifier model.score 0.5372727272727272
# XGBClassifier : [0.05775603 0.08979715 0.06077373 0.066026   0.06015197 0.06579991
#  0.06395647 0.0578346  0.06158298 0.06858748 0.15537561 0.19235805]