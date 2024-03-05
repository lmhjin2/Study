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

# parameters = {
#     'n_estimators' : [100,200,300,400,500],
#     'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1],
#     'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'gamma' : [0, 1, 2],
#     'lambda' : [0, 0.1, 0.01],
#     'alpha' : [0, 0.1, 0.01]
# }
parameters = {'n_estimators':1000,
              'learning_rate': 0.01,
              'max_depth':3,
              'gamma':0,
              'min_child_weight':0,
              'subsample':0.4,  # dropout개념과 비슷
              'colsample_bytree':0.8,
              'colsample_bylevel':0.7,
              'colsample_bynode':1,
              'reg_alpha': 0,
              'reg_lamda': 1,
              'random_state': 3377,
              'verbose' :0
              }

#2 model
model = XGBRegressor(random_state=0)
model.set_params(early_stopping_rounds = 10, **parameters)

#3 compile train
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose= 0 )
#4 predict, test
results = model.score(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('model.score:', results)
print('r2:', results)

# #4
# y_submit = model.predict(test_csv)

# submission_csv['count']=y_submit
# submission_csv.to_csv(path+"submission_0215.csv",index=False)

# 선택된 특성 수: 9
# 컬럼 줄인 XGBRegressor 의 정확도: 0.9744499817510559

# 떨어짐
# 최적의 파라미터 :  {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.2, 'lambda': 0, 'gamma': 0, 'alpha': 0.1}
# best_score : 0.7257779084756383
# 최적 튠 R2: 0.7748756550354514
# r2: [0.69522723 0.58364515 0.65019735]
#  평균 r2: 0.643
# model.score: 0.7748756550354514
# r2: 0.7748756550354514
# 걸린시간: 2.34 초

####################################################################################
feature_importances_list = list(model.feature_importances_)
# print(feature_importances_list)
feature_importances_list_sorted = sorted(feature_importances_list)
# print(feature_importances_list_sorted)  # 0번이제일 낮고 29번이 제일높음 (총 컬럼 30개)
drop_feature_idx_list = [feature_importances_list.index(feature) for feature in feature_importances_list_sorted]
print(drop_feature_idx_list)

result_dict = {}
for i in range(len(drop_feature_idx_list)): # 1바퀴에 1개, 마지막 바퀴에 29개 지우기, len -1은 30개 다지우면 안돼서
    drop_idx = drop_feature_idx_list[:i] # +1을 해준건 첫바퀴에 한개를 지워야 해서. 30개 시작 하고싶으면 i 만쓰고 위에 -1 지워주면됨
    new_x_train = np.delete(x_train, drop_idx, axis = 1)
    new_x_test = np.delete(x_test, drop_idx, axis=1)
    print(new_x_train.shape, new_x_test.shape)
    
    model2 = XGBRegressor()
    model2.set_params(early_stopping_rounds = 10, **parameters)
    model2.fit(new_x_train,y_train,
          eval_set=[(new_x_train,y_train), (new_x_test,y_test)],
          verbose=0
          )
    new_result = model2.score(new_x_test,y_test)
    print(f"{i}개 컬럼이 삭제되었을 때 Score: ",new_result)
    result_dict[i] = new_result - results    # 그대로 보면 숫자가 비슷해서 구분하기 힘들기에 얼마나 변했는지 체크
    
    
print(result_dict)

# model.score: 0.7656069962411634
# r2: 0.7656069962411634
# 1: 0.008625720812430182
