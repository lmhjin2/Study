import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_squared_log_error
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor,StackingRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators
from xgboost import XGBRegressor
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape)        # (442, 10)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 1)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# pca = PCA(n_components=8)
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


# best_score : 0.4511509586605877
# T r2: 0.3422603227510965
# F r2: 0.12475664605508174

# VS r2: 0.24160631151890655

# model.score :  0.3119202938666691
# 스태킹 R2 : 0.3119202938666691

# {'target': 0.4317382379599407, 'params': {'colsample_bytree': 0.5904955707290681, 
# 'learning_rate': 0.29109734445446084, 'max_bin': 447.7447895697852, 
# 'max_depth': 9.665620387507126, 'min_child_samples': 194.124275378605, 
# 'min_child_weight': 17.063253642015447, 'num_leaves': 25.643225406497734, 
# 'reg_alpha': 44.06392410483676, 'reg_lambda': 0.396022238383653, 'subsample': 0.5253296174165291}}
# 100 번 도는데 걸린시간 :  11.05 초