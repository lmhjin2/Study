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
from sklearn.ensemble import RandomForestClassifier
import time as tm

#1
datasets = load_wine()
x = datasets.data
y = datasets.target

x = x.astype(np.float32)
y = y.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split': [2,3,5,10]},
    {'min_samples_split': [2,3,5,10]},
    {'n_jobs':[-1,2,4], 'min_samples_split': [2,3,5,10]}
    ]

#2 모델

model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv = kfold,
                    verbose=1, 
                    refit = True, 
                    n_jobs=-1,     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
                    factor = 2,
                    min_resources= 13
                    )
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print("최적의 매개변수 : ", model.best_estimator_)
    # 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
    # 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100}
print('best_score :', model.best_score_)
    # best_score : 0.9785714285714286
print('model.score :', model.score(x_test, y_test))
    # model.score : 1.0

y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test,y_predict))
    # accuracy_score: 1.0
y_pred_best = model.best_estimator_.predict(x_test)
            # 최적의 매개변수.predict(x_test)
print('최적 튠 ACC:', accuracy_score(y_test,y_pred_best))
    # 최적 튠 ACC: 1.0

print('걸린시간:', np.round(end_time - strat_time, 2), '초')

# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3)
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100}
# best_score : 0.9785714285714286
# model.score : 1.0
# accuracy_score: 1.0
# 최적 튠 ACC: 1.0
# 걸린시간: 45.92 초

# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=10, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 10}
# best_score : 0.9785714285714286
# model.score : 1.0
# accuracy_score: 1.0
# 최적 튠 ACC: 1.0
# 걸린시간: 2.57 초

# ----------
# iter: 3
# n_candidates: 8
# n_resources: 104
# Fitting 10 folds for each of 8 candidates, totalling 80 fits
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=4)
# 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': 4}
# best_score : 0.96
# model.score : 1.0
# accuracy_score: 1.0
# 최적 튠 ACC: 1.0
# 걸린시간: 8.04 초