# https://dacon.io/competitions/open/235610/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings('ignore')

path = "c:/_data/dacon/wine/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")


train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)

scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x.shape)    # (5497, 12)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

#2. model
start_t = tm.time()
bayesian_params = {
    'learning_rate' : (0.001, 1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : ( -0.001, 10),
    'reg_alpha' : (0.01, 50)
}
def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),      # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),    # min은 최소값을 반환, max는 최대값을 반환. 0과 1사이로 숫자를 뽑음
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),   # 10이상의 정수
        'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
        'reg_alpha' : reg_alpha,       
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

bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds=bayesian_params,
    random_state=777
)

n_iter = 100
bay.maximize(init_points=5, n_iter=n_iter)

print(bay.max)
end_t = tm.time()

print(n_iter, "번 도는데 걸린시간 : ", np.round(end_t - start_t, 2), '초')

# model.score : 0.6454545454545455
# T acc:  0.6572727272727272
# F acc:  0.65

# VS acc:  0.6672727272727272
# VH acc:  0.6718181818181819

# model.score :  0.6627272727272727
# 스태킹 ACC : 0.6627272727272727

# {'target': 0.6509090909090909, 'params': {'colsample_bytree': 0.9335933967535406, 
# 'learning_rate': 0.509342268375129, 'max_bin': 300.25927970706056, 'max_depth': 9.780341351368206, 
# 'min_child_samples': 26.672572374948057, 'min_child_weight': 3.118414204435079, 
# 'num_leaves': 37.377718974537345, 'reg_alpha': 2.6744591200430534, 
# 'reg_lambda': 4.968920178840072, 'subsample': 0.601309572256216}}
# 100 번 도는데 걸린시간 :  26.78 초