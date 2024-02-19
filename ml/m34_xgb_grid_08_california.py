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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import  PCA

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
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)
parameters = {
    'n_estimators' : [100,200,300,400,500],
    'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1],
    'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'gamma' : [0, 1, 2],
    'lambda' : [0, 0.1, 0.01],
    'alpha' : [0, 0.1, 0.01]
}
#2
models = [DecisionTreeRegressor(random_state= 0), RandomForestRegressor(random_state= 0),
          GradientBoostingRegressor(random_state= 0), XGBRegressor(random_state= 0)]
#2 model
xgb = XGBRegressor(random_state=0)
model = RandomizedSearchCV(xgb, parameters, cv = kfold, refit=True, n_jobs= 22 )

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
scores = cross_val_score(model, x_test, y_test, cv=kfold)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
print('best_score :', model.best_score_)
print('최적 튠 R2:', r2_score(y_test,y_pred_best))
print('r2:', scores, "\n 평균 r2:", round(np.mean(scores), 4))
print('model.score:', results)
print('r2:', results)
print('걸린시간:', np.round(end_time - start_time, 2), '초')



# XGBRegressor model.score 0.735695874046558
# XGBRegressor : [0.04885179 0.04130459 0.03545461 0.58793354 0.0550629  0.09798978
#  0.13340269]

# 선택된 특성 수: 7
# 컬럼 줄인 XGBRegressor 의 정확도: 0.7356383802720532