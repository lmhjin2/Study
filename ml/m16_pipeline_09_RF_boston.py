# restore_best_weights 와
# save_best_only 에 대한 고찰

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import time as tm
from sklearn.pipeline import make_pipeline
#1
datasets = load_boston()

x = datasets.data
y = datasets.target

# 라벨 인코딩. StratifiedKFold 할때만 필요
label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)
# 라벨 인코딩. StratifiedKFold 할때만 필요
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

n_splits =  5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'n_estimators':[100,200,300], 'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split': [2,3,5,10]},
    {'min_samples_split': [2,3,5,10]}
    ]

#2 모델
model = make_pipeline(MaxAbsScaler(), RandomForestRegressor(min_samples_split=5, n_jobs=4))
# model = HalvingGridSearchCV(RandomForestRegressor(), parameters, 
#                     # cv = kfold,
#                     cv = 5,
#                     verbose=1, 
#                     refit = True, 
#                     # n_jobs=-1,     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
#                     # factor=2,
#                     # min_resources=20,
#                     )
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.7407822210859201

y_predict = model.predict(x_test)
print('r2_score:', r2_score(y_test,y_predict))
    # r2_score: 0.7407822210859201

print('걸린시간:', np.round(end_time - strat_time, 2), '초')
# 걸린시간: 65.34 초

# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=5, n_jobs=4)
# 최적의 파라미터 :  {'n_jobs': 4, 'min_samples_split': 5}
# best_score : 0.8702929092828047
# model.score : 0.7465215453931853
# r2_score: 0.7465215453931853
# 최적 튠 R2: 0.7465215453931853
# 걸린시간: 9.13 초

# pipeline
# model.score : 0.7422874307502669
# r2_score: 0.7422874307502669
# 걸린시간: 0.07 초