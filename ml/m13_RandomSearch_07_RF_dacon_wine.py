# https://dacon.io/competitions/open/235610/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators

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

scaler = MaxAbsScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

n_splits =  5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'n_estimators':[100,200,300], 'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split': [2,3,5,10]},
    {'min_samples_split': [2,3,5,10]}
    ]

#2 모델

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold,
                    # verbose=1, 
                    refit = True, 
                    # n_jobs=-1     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
                    n_iter=100
                    )
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print("최적의 매개변수 : ", model.best_estimator_)
    # 최적의 매개변수 :  RandomForestClassifier(n_jobs=4)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
    # 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 4}
print('best_score :', model.best_score_)
    # best_score : 0.6654431559329054
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.6754545454545454

y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test,y_predict))
    # accuracy_score: 0.6754545454545454
y_pred_best = model.best_estimator_.predict(x_test)
            # 최적의 매개변수.predict(x_test)
print('최적 튠 ACC:', accuracy_score(y_test,y_pred_best))
    # 최적 튠 ACC: 0.6754545454545454

print('걸린시간:', np.round(end_time - strat_time, 2), '초')

# 최적의 매개변수 :  RandomForestClassifier(n_jobs=4)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 4}
# best_score : 0.6654431559329054
# model.score : 0.6754545454545454
# accuracy_score: 0.6754545454545454
# 최적 튠 ACC: 0.6754545454545454
# 걸린시간: 272.54 초

# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3)
# 최적의 파라미터 :  {'min_samples_split': 3}
# best_score : 0.6602197745371806
# model.score : 0.6790909090909091
# accuracy_score: 0.6790909090909091
# 최적 튠 ACC: 0.6790909090909091
# 걸린시간: 162.35 초