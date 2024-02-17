# https://dacon.io/competitions/official/236214/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA

np.set_printoptions(suppress=True)
path = "c:/_data/dacon/dechul/"
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

le_work_period = LabelEncoder() 
le_work_period.fit(train_csv['근로기간'])
train_csv['근로기간'] = le_work_period.transform(train_csv['근로기간'])
test_csv['근로기간'] = le_work_period.transform(test_csv['근로기간'])

le_purpose = LabelEncoder()
test_csv.iloc[34486,7] = '이사'     # 결혼 -> 이사 로 임의로 바꿈
le_purpose.fit(train_csv['대출목적'])
train_csv['대출목적'] = le_purpose.transform(train_csv['대출목적'])
test_csv['대출목적'] = le_purpose.transform(test_csv['대출목적'])

le_own = LabelEncoder()
le_own.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = le_own.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = le_own.transform(test_csv['주택소유상태'])

le_loan_period = LabelEncoder()
le_loan_period.fit(train_csv['대출기간'])
train_csv['대출기간'] = le_loan_period.transform(train_csv['대출기간'])
test_csv['대출기간'] = le_loan_period.transform(test_csv['대출기간'])

le_grade = LabelEncoder()
le_grade.fit(train_csv['대출등급'])
train_csv['대출등급'] = le_grade.transform(train_csv['대출등급'])

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

# print(x.shape, y.shape) # 13 columns

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
        series = series.interpolate()   # 결측치 interpolate()로 채우기
        data[label] = series    # 원래 위치에 덮어쓰기
        
    # data = data.fillna(data.ffill())
    # data = data.fillna(data.bfill())  
    return data
# print(x.isna().sum())
x = fit_outlier(x)
# print(x.isna().sum())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size = 0.18, random_state = 1785 )
#2 모델
model = XGBClassifier()
# 3
model.fit(x_train,y_train)
# 4 
results = model.score(x_test,y_test)
print("model.score",results)
y_predict = model.predict(x_test)
print('r2', accuracy_score(y_test, y_predict))
print('f1', f1_score(y_test, y_predict, average='macro'))

# y_submit = model.predict(test_csv)

# submission_csv['count']=y_submit
# submission_csv.to_csv(path+"submission_0217.csv",index=False)

# model.score 0.8152079847689379
# r2 0.8152079847689379
# f1 0.6699150043102325

