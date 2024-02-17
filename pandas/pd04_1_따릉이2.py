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

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path+"train.csv",index_col=['id'])
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"submission.csv")

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
# 함수 생성
def fit_outlier(data):  
    data = pd.DataFrame(data)
    for label in data:
        series = data[label]        # data의 label이라는 컬럼의 데이터를 series에 담음
        q1 = series.quantile(0.25)  # q1 = 25퍼센트 지점  
        q3 = series.quantile(0.75)  # q3 = 75퍼센트 지점
        iqr = q3 - q1
        upper_bound = q3 + (iqr * 1.5)     # 이상치 범위 설정
        lower_bound = q1 - (iqr * 1.5)
        
        series[series > upper_bound] = np.nan   # series안에 이상치들 전부 np.nan(결측치) 처리
        series[series < lower_bound] = np.nan
        print(series.isna().sum())      # series 안에 결측치 갯수
        # series = series.interpolate()   # 결측치 interpolate()로 채우기
        data[label] = series    # 원래 위치에 덮어쓰기
        
    # data = data.fillna(data.ffill())
    # data = data.fillna(data.bfill())  
    return data
# print(x.isna().sum())
x = fit_outlier(x)
print(x.isna().sum())

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
imputer = KNNImputer()  # 데이터셋 내에서 가장 비슷한 데이터와 비슷하게 결측치 바꿔줌
x1 = imputer.fit_transform(x)   # x에서 이상치->결측치 만들고 knnimputer 적용
# print(x1.isnan().sum())
print(np.isnan(x1).sum())       # 결측치 확인

test_csv = test_csv.fillna(test_csv.mean()) # 결측치에 평균치넣기

x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, shuffle=True, random_state= 0)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)
np.set_printoptions(suppress=True)

#2
model = XGBRegressor(random_state= 0)

#3 
model.fit(x_train, y_train)

# 4
result = model.score(x_test,y_test)
print("result",result)
y_predict = model.predict(x_test)
print('r2', r2_score(y_test, y_predict))


y_submit = model.predict(test_csv)

submission_csv['count']=y_submit
submission_csv.to_csv(path+"submission_0217.csv",index=False)

# result 0.790588400462531
# r2 0.790588400462531