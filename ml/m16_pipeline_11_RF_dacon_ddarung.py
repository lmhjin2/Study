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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import time as tm
from sklearn.pipeline import make_pipeline

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

parameters = [
    {'n_estimators':[100,200,300], 'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split': [2,3,5,10]},
    {'min_samples_split': [2,3,5,10]}
    ]

#2
model = make_pipeline(StandardScaler(), RandomForestRegressor())
# model = HalvingGridSearchCV(RandomForestRegressor(), parameters, cv = kfold,
#                     verbose=1, 
#                     refit = True, 
#                     # n_jobs=-1,     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
#                     # factor=2,
#                     # min_resources=20,
#                     )
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.7883677786831836
y_predict = model.predict(x_test)
print('r2_score:', r2_score(y_test,y_predict))
    # r2_score: 0.7883677786831836

print('걸린시간:', np.round(end_time - strat_time, 2), '초')
# 걸린시간: 103.13 초

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
