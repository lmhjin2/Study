# https://dacon.io/competitions/open/235576/mysubmission
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error, mean_absolute_error
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor,\
    VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings('ignore')


plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.dropna()  # 결측치 드랍.
test_csv = test_csv.fillna(test_csv.mean()) # 결측치에 평균치넣기
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
# print(x.shape, y.shape)       # (1328, 10)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)
# pca = PCA(n_components=9)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)
# test_csv = pca.transform(test_csv)


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


# 선택된 특성 수: 9
# 컬럼 줄인 XGBRegressor 의 정확도: 0.9744499817510559

# T r2: 0.8063415747782201
# F r2: 0.7949821318380946

# VS r2: 0.8033612950258426

# model.score :  0.7720671171017182
# 스태킹 R2 : 0.7720671171017182

# {'target': 0.8145947046134788, 'params': {'colsample_bytree': 0.8846827876574986, 
# 'learning_rate': 0.08588171777293258, 'max_bin': 28.195474161824663, 
# 'max_depth': 6.7375819649546616, 'min_child_samples': 157.45142174846518, 
# 'min_child_weight': 8.424642631204682, 'num_leaves': 25.466552795574117, 
# 'reg_alpha': 11.219742144714044, 'reg_lambda': 7.365554659170791, 'subsample': 0.8526075414053081}}
# 100 번 도는데 걸린시간 :  12.01 초

# Best accuracy: 0.7968538428
# {'colsample_bytree': 0.5543852804372733, 'learning_rate': 0.9354705629059283,
# 'max_bin': 304.0, 'max_depth': 9.0, 'min_child_samples': 156.0, 'min_child_weight': 30.0, 
# 'num_leaves': 40.0, 'reg_alpha': 2.5579538642531987, 'reg_lambda': 9.674065092477512,
# 'subsample': 0.5506226585472458}
# 100 번 도는데 걸린시간 :  2.27 초

