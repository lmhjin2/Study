import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import time as tm
import warnings
warnings.filterwarnings('ignore')

#1. data
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
    stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from bayes_opt import BayesianOptimization

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
    model = XGBClassifier(**params, n_jobs=-1)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train),(x_test, y_test)],
            #   eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50,
              )
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
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

# model.score :  1.0
# 스태킹 ACC : 1.0

# {'colsample_bytree': 0.5300418778765242, 'learning_rate': 0.12672595272780718, 'max_bin': 313.0, 
# 'max_depth': 6.0, 'min_child_samples': 135.0, 'min_child_weight': 4.0, 'num_leaves': 34.0, 
# 'reg_alpha': 0.48385618387130336, 'reg_lambda': 9.913429190547106, 'subsample': 0.9276299549911071}

# {'target': 0.9912280701754386, 'params': {'colsample_bytree': 0.8876413839415261, 'learning_rate': 0.5286769460422218, 
# 'max_bin': 241.67494439003562, 'max_depth': 6.354509659610262, 'min_child_samples': 178.23154404267854, 
# 'min_child_weight': 2.277586113115521, 'num_leaves': 27.601344619311064, 
# 'reg_alpha': 2.2403347912292975, 'reg_lambda': 7.286498703544207, 'subsample': 0.6447047036574494}}
# 500 번 도는데 걸린시간 :  224.1 초