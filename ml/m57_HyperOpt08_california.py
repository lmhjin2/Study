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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor,\
    VotingRegressor, StackingRegressor
from sklearn.preprocessing import LabelEncoder
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.decomposition import  PCA

import warnings

warnings.filterwarnings('ignore')


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
# print(x.shape, y.shape)   # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# pca = PCA(n_components=7)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

from hyperopt import hp, fmin,tpe, Trials, STATUS_OK
#2. model
start_t = tm.time()
search_space = {
    'learning_rate' : hp.uniform('learning_rate',0.001, 1),
    'max_depth' : hp.quniform('max_depth', 3, 10, 1),
    'num_leaves' : hp.quniform('num_leaves', 24, 40, 1),
    'min_child_samples' : hp.quniform('min_child_samples', 10, 200, 1),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 50, 1),
    'subsample' : hp.uniform('subsample', 0.5, 1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
    'max_bin' : hp.quniform('max_bin', 9, 500, 1),
    'reg_lambda' : hp.uniform('reg_lambda',  -0.001, 10),
    'reg_alpha' : hp.uniform('reg_alpha', 0.01, 50)
}
def xgb_hamsu(search_space):
    params = {
        'n_estimators' : 100,
        'learning_rate' : search_space['learning_rate'],
        'max_depth' : int(search_space['max_depth']),      # 무조건 정수형
        'num_leaves' : int(search_space['num_leaves']),
        'min_child_samples' : int(search_space['min_child_samples']),
        'min_child_weight' : int(search_space['min_child_weight']),
        'subsample' : max(min(search_space['subsample'], 1), 0),    # min은 최소값을 반환, max는 최대값을 반환. 0과 1사이로 숫자를 뽑음
        'colsample_bytree' : search_space['colsample_bytree'],
        'max_bin' : max(int(search_space['max_bin']), 10),   # 10이상의 정수
        'reg_lambda' : max(search_space['reg_lambda'], 0),          # 무조건 양수만
        'reg_alpha' : search_space['reg_alpha'],       
    }
    model = XGBRegressor(**params, n_jobs=-1)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train),(x_test, y_test)],
            #   eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50,
              )
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results


trial_val = Trials()
n_iter = 100

best = fmin(
    fn = xgb_hamsu,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)
best_trial = sorted(trial_val.trials, key=lambda x: x["result"]["loss"], reverse=True)[0]
best_acc = best_trial["result"]["loss"]

print(f"Best accuracy: {best_acc:.10f}")
end_t = tm.time()
print(best)
print(n_iter, "번 도는데 걸린시간 : ", np.round(end_t - start_t, 2), '초')

# model.score: 0.8534372129491651
# T r2: 0.858944147480187
# F r2: 0.8442317659339414

# VS r2: 0.8408760327999003

# model.score :  0.8411648807974154
# 스태킹 R2 : 0.8411648807974154

# {'target': 0.856344020803159, 'params': {'colsample_bytree': 0.6291389181988791, 
# 'learning_rate': 0.14761741781225676, 'max_bin': 209.88471540907602, 
# 'max_depth': 9.812916670055387, 'min_child_samples': 130.4778769658467, 
# 'min_child_weight': 32.13475173953371, 'num_leaves': 34.60822604995035, 
# 'reg_alpha': 9.766142119394104, 'reg_lambda': 2.3984308120293125, 'subsample': 0.5722897072370908}}
# 100 번 도는데 걸린시간 :  21.49 초

# Best accuracy: 0.8553468604
# {'colsample_bytree': 0.8942888156460665, 'learning_rate': 0.0014227149193139369, 
# 'max_bin': 245.0, 'max_depth': 4.0, 'min_child_samples': 25.0, 'min_child_weight': 50.0, 
# 'num_leaves': 29.0, 'reg_alpha': 44.9655920402184, 'reg_lambda': 0.5361991936322211, 'subsample': 0.9410233782474909}
# 100 번 도는데 걸린시간 :  5.83 초