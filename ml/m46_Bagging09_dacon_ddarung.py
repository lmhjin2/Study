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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LogisticRegression
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


#2 model
xgb = XGBRegressor()
# xgb = LogisticRegression()
model = BaggingRegressor(xgb,
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=777,
                          bootstrap=True, # 기본값, 데이터 중복 허용. (샘플링)
                        #   bootstrap=False,    # 중복 허용 안함
                          
                          )

#3 compile train
model.fit(x_train, y_train)

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

# r2: 0.7748756550354514
# r2: 0.7656069962411634

# T r2: 0.8063415747782201
# F r2: 0.7949821318380946

