 # https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_error
import time as tm
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

#1
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path +"train.csv", index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

# scaler = MinMaxScaler()
# scaler.fit(x)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

parameters = [
    {'RF__n_estimators':[100,200,300], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split': [2,3,5,10]},
    {'RF__min_samples_split': [2,3,5,10]}
    ]
#2
pipe = Pipeline([('MinMax', MinMaxScaler()), ('RF', RandomForestRegressor(max_depth=10, min_samples_leaf=3))])
model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.3537637566261046
y_predict = model.predict(x_test)
print('r2_score:', r2_score(y_test,y_predict))
    # r2_score: 0.3537637566261046

print('걸린시간:', np.round(end_time - strat_time, 2), '초')
# 걸린시간: 494.15 초

# print(pd.DataFrame(model.cv_results_).transpose()) # 잘 안보이니까 dataframe에 담아서 따로 열던가 csv파일로 만들어서 보던가

# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_leaf=3)
# 최적의 파라미터 :  {'n_estimators': 100, 'min_samples_leaf': 3, 'max_depth': 10}
# best_score : 0.3588859634566727
# model.score : 0.3540758610819569
# r2_score: 0.3540758610819569
# 최적 튠 R2: 0.3540758610819569
# 걸린시간: 24.64 초

# pipeline
# model.score : 0.35409097313020055
# r2_score: 0.35409097313020055
# 걸린시간: 0.59 초
