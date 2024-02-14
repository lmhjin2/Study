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
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.utils import all_estimators
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

n_splits =  4
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = RobustScaler()

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2
parameters = [
    {'RF__n_estimators':[100,200,300], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split': [2,3,5,10]},
    {'RF__min_samples_split': [2,3,5,10]}
    ]

#2 모델
pipe = Pipeline([('Robust', RobustScaler()), ('RF', RandomForestClassifier(max_depth=12, min_samples_leaf=5, n_estimators=300))])
model = GridSearchCV(pipe, parameters, cv=2, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.9553539925819471

y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test,y_predict))
    # 최적 튠 ACC: 0.9553539925819471

print('걸린시간:', np.round(end_time - strat_time, 2), '초')
    # 걸린시간: 21265.71초

# 최적의 매개변수 :  RandomForestClassifier(n_jobs=2)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 2}
# best_score : 0.9531721667795265
# model.score : 0.9553539925819471
# accuracy_score: 0.9553539925819471
# 최적 튠 ACC: 0.9553539925819471
# 걸린시간: 21265.71초

# iter: 4
# n_candidates: 2
# n_resources: 464778
# Fitting 3 folds for each of 2 candidates, totalling 6 fits
# 최적의 매개변수 :  RandomForestClassifier()
# 최적의 파라미터 :  {'min_samples_split': 2}
# best_score : 0.9459072756467158
# model.score : 0.9549925561302204
# accuracy_score: 0.9549925561302204
# 최적 튠 ACC: 0.9549925561302204
# 걸린시간: 170.87 초

# pipeline
# model.score : 0.9541922325585398
# accuracy_score: 0.9541922325585398
# 걸린시간: 30.7 초

# Pipeline
# model.score : 0.784024508833679
# accuracy_score: 0.784024508833679
# 걸린시간: 6750.72 초

