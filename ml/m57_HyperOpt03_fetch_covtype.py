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
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier,VotingClassifier, StackingClassifier
from sklearn.utils import all_estimators
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings('ignore')

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(x.shape)    # (581012, 54)
scaler = StandardScaler()
x = scaler.fit_transform(x)

label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

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

# model.score :  0.9614123559632711
# 스태킹 ACC : 0.9614123559632711

# {'target': 0.9589167233203962, 'params': {'colsample_bytree': 0.9335933967535406, 
# 'learning_rate': 0.509342268375129, 'max_bin': 300.25927970706056, 'max_depth': 9.780341351368206, 
# 'min_child_samples': 26.672572374948057, 'min_child_weight': 3.118414204435079, 'num_leaves': 37.377718974537345, 
# 'reg_alpha': 2.6744591200430534, 'reg_lambda': 4.968920178840072, 'subsample': 0.601309572256216}}
# 100 번 도는데 걸린시간 :  913.21 초

