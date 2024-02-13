# https://dacon.io/competitions/official/236214/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.utils import all_estimators

path = "c:/_data/dacon/dechul/"
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')


le_work_period = LabelEncoder() 
le_work_period.fit(train_csv['근로기간'])
train_csv['근로기간'] = le_work_period.transform(train_csv['근로기간'])
test_csv['근로기간'] = le_work_period.transform(test_csv['근로기간'])

le_grade = LabelEncoder()
le_grade.fit(train_csv['대출등급'])
train_csv['대출등급'] = le_grade.transform(train_csv['대출등급'])

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

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size = 0.18, random_state = 1818 )
# 1785 / 1818 / 

from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split': [2,3,5,10]},
    {'min_samples_split': [2,3,5,10]},
    {'n_jobs':[-1,2,4], 'min_samples_split': [2,3,5,10]}
    ]

#2 모델

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold,
                    # verbose=1, 
                    refit = True, 
                    # n_jobs=-1     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
                    )
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print("최적의 매개변수 : ", model.best_estimator_)
    # 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=2)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
    # 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': 2}
print('best_score :', model.best_score_)
    # best_score : 0.8032572758895912
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.8046500894247967

y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test,y_predict))
    # accuracy_score: 0.8046500894247967
y_pred_best = model.best_estimator_.predict(x_test)
            # 최적의 매개변수.predict(x_test)
print('최적 튠 ACC:', accuracy_score(y_test,y_pred_best))
    # 최적 튠 ACC: 0.8046500894247967

print('걸린시간:', np.round(end_time - strat_time, 2), '초')
# 걸린시간: 2783.57 초


#3
# import datetime

# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")   # 월일_시분

# path1 = "c:/_data/_save/MCP/k28/11/"
# filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
# filepath = "".join([path1, 'k28_', date, '_1_', filename])


#4
# y_submit = model.predict(test_csv)

# y_submit = le_grade.inverse_transform(y_submit)

# submission_csv['대출등급'] = y_submit
# submission_csv.to_csv(path + "submission_0208_1.csv", index=False)
# https://dacon.io/competitions/official/236214/mysubmission


# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=2)
# 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': 2}
# best_score : 0.8032572758895912
# model.score : 0.8046500894247967
# accuracy_score: 0.8046500894247967
# 최적 튠 ACC: 0.8046500894247967
# 걸린시간: 2783.57 초

# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=5, n_jobs=4)
# 최적의 파라미터 :  {'n_jobs': 4, 'min_samples_split': 5}
# best_score : 0.7985080768216408
# model.score : 0.8015346448970172
# accuracy_score: 0.8015346448970172
# 최적 튠 ACC: 0.8015346448970172
# 걸린시간: 177.19 초
