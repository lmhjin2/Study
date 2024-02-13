import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
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
from sklearn.pipeline import make_pipeline, Pipeline

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 1)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

n_splits =  5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'RF__n_estimators':[100,200,300], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split': [2,3,5,10]},
    {'RF__min_samples_split': [2,3,5,10]}
    ]
#2
pipe = Pipeline([('MinMax', MinMaxScaler()), ('RF', RandomForestClassifier(max_depth=10, min_samples_leaf=5))])
model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

# model = make_pipeline(MaxAbsScaler(), RandomForestClassifier(max_depth=10, min_samples_leaf=5))
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.3158090196960497
y_predict = model.predict(x_test)
print('r2_score:', r2_score(y_test,y_predict))
    # r2_score: 0.3158090196960497

print('걸린시간:', np.round(end_time - strat_time, 2), '초')

# ----------
# iter: 3
# n_candidates: 3
# n_resources: 351
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# c:\Users\AIA\anaconda3\envs\tf290gpu\lib\site-packages\sklearn\model_selection\_split.py:684: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
#   warnings.warn(
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_leaf=5)
# 최적의 파라미터 :  {'max_depth': 10, 'min_samples_leaf': 5}
# best_score : 0.46630723859494444
# model.score : 0.33819940383645386
# r2_score: 0.33819940383645386
# 최적 튠 R2: 0.33819940383645386
# 걸린시간: 4.1 초

# pipeline
# model.score : 0.0
# r2_score: -0.188934701837024
# 걸린시간: 0.12 초

# Pipeline
# model.score : 0.0
# r2_score: -0.41317010675994714
# 걸린시간: 58.6 초