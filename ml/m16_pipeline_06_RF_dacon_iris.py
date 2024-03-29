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
from sklearn.ensemble import RandomForestClassifier
import time as tm
from sklearn.pipeline import make_pipeline

#1
path = "c:/_data/dacon/iris/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv.drop(['species'], axis = 1)
y = train_csv['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

parameters = [
    {'n_estimators':[100,200,300], 'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split': [2,3,5,10]},
    {'min_samples_split': [2,3,5,10]}
    ]

#2 모델
model = make_pipeline(MaxAbsScaler(), RandomForestClassifier(min_samples_split = 2))
# model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv = kfold,
#                     verbose=1, 
#                     refit = True, 
#                     n_jobs=-1,     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
#                     factor = 2, 
#                     min_resources= 20,
#                     )
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.9166666666666666

y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test,y_predict))
    # accuracy_score: 0.9166666666666666

print('걸린시간:', np.round(end_time - strat_time, 2), '초')
# 걸린시간: 44.38 초

# 최적의 매개변수 :  RandomForestClassifier()
# 최적의 파라미터 :  {'min_samples_split': 2}
# best_score : 0.9266666666666665
# model.score : 0.9583333333333334
# accuracy_score: 0.9583333333333334
# 최적 튠 ACC: 0.9583333333333334
# 걸린시간: 7.27 초

# ----------
# iter: 2
# n_candidates: 21
# n_resources: 80
# Fitting 10 folds for each of 21 candidates, totalling 210 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=8, min_samples_leaf=5)
# 최적의 파라미터 :  {'max_depth': 8, 'min_samples_leaf': 5}
# best_score : 0.9589285714285716
# model.score : 0.9166666666666666
# accuracy_score: 0.9166666666666666
# 최적 튠 ACC: 0.9166666666666666
# 걸린시간: 12.86 초

# pipeline
# model.score : 0.9583333333333334
# accuracy_score: 0.9583333333333334
# 걸린시간: 0.04 초