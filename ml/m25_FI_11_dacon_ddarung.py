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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = StandardScaler()

# scaler.fit(x)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

#2
models = [DecisionTreeRegressor(random_state= 0), RandomForestRegressor(random_state= 0),
          GradientBoostingRegressor(random_state= 0), XGBRegressor(random_state= 0)]

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "r2 score", r2_score(y_predict, y_test))
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_, end='\n\n')
        ## type(model).__name__ == 모델 이름만 뽑기
        # end = '\n\n' == print('\n') 한줄 추가
    except Exception as e:
        print("에러:", e)
        continue


# #4
# y_submit = model.predict(test_csv)

# submission_csv['count']=y_submit
# submission_csv.to_csv(path+"submission_0131.csv",index=False)

# 최적의 매개변수 :  RandomForestRegressor()
# 최적의 파라미터 :  {'min_samples_split': 2}
# best_score : 0.7652611100092658
# model.score : 0.7904932568912093
# r2_score: 0.7904932568912093
# 최적 튠 R2: 0.7904932568912093
# 걸린시간: 203.43 초

# pipeline
# model.score : 0.7903707059202136
# r2_score: 0.7903707059202136
# 걸린시간: 0.24 초


# Pipeline
# model.score : 0.7953029668997788
# r2_score: 0.7953029668997788
# 걸린시간: 0.82 초

# DecisionTreeRegressor r2 score 0.6381858402876152
# DecisionTreeRegressor model.score 0.5930602764010395
# DecisionTreeRegressor : [0.58366329 0.19587694 0.00651891 0.02859404 0.04551763 0.05413454
#  0.0287678  0.03429513 0.02263172]

# RandomForestRegressor r2 score 0.7378440612432596
# RandomForestRegressor model.score 0.7906523416159639
# RandomForestRegressor : [0.59509245 0.1776177  0.01620756 0.03118662 0.03771075 0.03792288
#  0.04163992 0.0390983  0.02352382]

# GradientBoostingRegressor r2 score 0.7089835742657045
# GradientBoostingRegressor model.score 0.7764621639045257
# GradientBoostingRegressor : [0.65092741 0.20562871 0.02037336 0.01067317 0.01459179 0.03031671
#  0.0271698  0.02988211 0.01043695]

# XGBRegressor r2 score 0.7666172493922796
# XGBRegressor model.score 0.7949821360117232
# XGBRegressor : [0.38300964 0.11773865 0.27954364 0.02146033 0.03986756 0.03715678
#  0.03737744 0.04594545 0.03790054]