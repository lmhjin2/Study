import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_squared_log_error, accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression, ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators
import time as tm

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 1)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits =  5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split': [2,3,5,10]},
    {'min_samples_split': [2,3,5,10]},
    {'n_jobs':[-1,2,4], 'min_samples_split': [2,3,5,10]}
    ]

#2
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv = kfold,
                    # verbose=1, 
                    refit = True, 
                    n_jobs=-1     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
                    )
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print("최적의 매개변수 : ", model.best_estimator_)
    # 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_leaf=10)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
    # 최적의 파라미터 :  {'max_depth': 10, 'min_samples_leaf': 10, 'n_estimators': 100}
print('best_score :', model.best_score_)
    # best_score : 0.47998972089553804
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.3158090196960497
y_predict = model.predict(x_test)
print('r2_score:', r2_score(y_test,y_predict))
    # r2_score: 0.3158090196960497
y_pred_best = model.best_estimator_.predict(x_test)
            # 최적의 매개변수.predict(x_test)
print('최적 튠 R2:', r2_score(y_test,y_pred_best))
    # 최적 튠 R2: 0.3158090196960497

print('걸린시간:', np.round(end_time - strat_time, 2), '초')

# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_leaf=10)
# 최적의 파라미터 :  {'max_depth': 10, 'min_samples_leaf': 10, 'n_estimators': 100}
# best_score : 0.47998972089553804
# model.score : 0.3158090196960497
# r2_score: 0.3158090196960497
# 최적 튠 R2: 0.3158090196960497
# 걸린시간: 35.68 초

# 최적의 매개변수 :  RandomForestRegressor(max_depth=6, min_samples_leaf=7)
# 최적의 파라미터 :  {'min_samples_leaf': 7, 'max_depth': 6}
# best_score : 0.4797092532593005
# model.score : 0.32371711920364776
# r2_score: 0.32371711920364776
# 최적 튠 R2: 0.32371711920364776
# 걸린시간: 2.34 초



