import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,mean_absolute_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline

plt.rcParams['font.family'],"Malgun Gothic"
plt.rcParams['axes.unicode_minus']=False

#1 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# 라벨 인코딩. StratifiedKFold 할때만 필요
label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)
# 라벨 인코딩. StratifiedKFold 할때만 필요
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

# scaler = MaxAbsScaler()

# scaler.fit(x)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'RF__n_estimators':[100,200,300], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split': [2,3,5,10]},
    {'RF__min_samples_split': [2,3,5,10]}
    ]

#2
pipe = Pipeline([('MaxAbs', MaxAbsScaler()), ('RF', RandomForestRegressor(n_jobs= 2, min_samples_split= 2))])
model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.807860614649011
y_predict = model.predict(x_test)
print('r2_score:', r2_score(y_test,y_predict))
    # r2_score: 0.807860614649011

print('걸린시간:', np.round(end_time - strat_time, 2), '초')
# 걸린시간: 1891.86 초

# 최적의 매개변수 :  RandomForestRegressor(n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 2}
# best_score : 0.8117560013468781
# model.score : 0.8110732528984922
# r2_score: 0.8110732528984922
# 최적 튠 R2: 0.8110732528984922
# 걸린시간: 156.97 초

# pipeline
# model.score : 0.8103574698657563
# r2_score: 0.8103574698657563
# 걸린시간: 2.81 초

# Pipeline
# model.score : 0.808833963853372
# r2_score: 0.808833963853372
# 걸린시간: 2790.1 초