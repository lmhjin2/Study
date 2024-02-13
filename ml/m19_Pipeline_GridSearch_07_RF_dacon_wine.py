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
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline, Pipeline

path = "c:/_data/dacon/wine/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")


train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

# scaler = MaxAbsScaler()

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

n_splits =  5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'RF__n_estimators':[100,200,300], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split': [2,3,5,10]},
    {'RF__min_samples_split': [2,3,5,10]}
    ]

#2 모델
pipe = Pipeline([('MinMax', MinMaxScaler()), ('RF', RandomForestClassifier(min_samples_split=3))])
model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.6754545454545454

y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test,y_predict))
    # accuracy_score: 0.6754545454545454

print('걸린시간:', np.round(end_time - strat_time, 2), '초')

# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3)
# 최적의 파라미터 :  {'min_samples_split': 3}
# best_score : 0.6602197745371806
# model.score : 0.6790909090909091
# accuracy_score: 0.6790909090909091
# 최적 튠 ACC: 0.6790909090909091
# 걸린시간: 162.35 초 


# pipeline
# model.score : 0.67
# accuracy_score: 0.67
# 걸린시간: 0.44 초


# Pipeline
# model.score : 0.6727272727272727
# accuracy_score: 0.6727272727272727
# 걸린시간: 161.17 초
