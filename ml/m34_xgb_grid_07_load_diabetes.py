import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_squared_log_error
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression, ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators
from xgboost import XGBRegressor
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

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
# n_splits = 2
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)
parameters = {
    'n_estimators' : [100,200,300,400,500],
    'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1],
    'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'gamma' : [0, 1, 2],
    'lambda' : [0, 0.1, 0.01],
    'alpha' : [0, 0.1, 0.01]
}

#2 model
xgb = XGBRegressor(random_state=0)
model = RandomizedSearchCV(xgb, parameters, refit=True, n_jobs= 22 )

#3 compile train
import time as tm
start_time = tm.time()
model.fit(x_train,y_train)
end_time = tm.time()
#4 predict, test
results = model.score(x_test,y_test)
y_predict = model.predict(x_test)
y_pred_best = model.best_estimator_.predict(x_test)
r2 = r2_score(y_test, y_predict)
# scores = cross_val_score(model, x_test, y_test, cv=kfold)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
print('best_score :', model.best_score_)
print('최적 튠 R2:', r2_score(y_test,y_pred_best))
# print('r2:', scores, "\n 평균 r2:", round(np.mean(scores), 4))
print('model.score:', results)
print('r2:', results)
print('걸린시간:', np.round(end_time - start_time, 2), '초')


# RandomForestRegressor model.score 0.3481006795397781
# RandomForestRegressor : [0.39823102 0.07971045 0.10852278 0.1940426  0.05468607 0.05358133
#  0.06559992 0.04562582]

# 선택된 특성 수: 8
# 컬럼 줄인 RandomForestRegressor 의 정확도: 0.3477970462470079

# kfold = 2
# 최적의 파라미터 :  {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.2, 'lambda': 0.1, 'gamma': 0, 'alpha': 0.01}
# best_score : 0.303825831577562
# 최적 튠 R2: 0.09665122500574186
# r2: [0.25786051 0.09556317]
#  평균 r2: 0.1767
# model.score: 0.09665122500574186
# r2: 0.09665122500574186
# 걸린시간: 2.2 초

# kfold 삭제
# 최적의 파라미터 :  {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.3, 'lambda': 0.01, 'gamma': 1, 'alpha': 0.1}
# best_score : 0.36270326802827824
# 최적 튠 R2: 0.19488003118421893
# model.score: 0.19488003118421893
# r2: 0.19488003118421893
# 걸린시간: 2.51 초