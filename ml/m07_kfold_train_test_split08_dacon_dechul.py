# https://dacon.io/competitions/official/236214/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier

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

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


#2
model = BaggingClassifier()

#3
# import datetime

# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")   # 월일_시분

# path1 = "c:/_data/_save/MCP/k28/11/"
# filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
# filepath = "".join([path1, 'k28_', date, '_1_', filename])

scores = cross_val_score(model, x, y, cv = kfold)
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

acc = accuracy_score(y_test, y_predict)
print('cross_val_precit acc:', acc)
#4
# y_submit = model.predict(test_csv)

# y_submit = le_grade.inverse_transform(y_submit)

# submission_csv['대출등급'] = y_submit
# submission_csv.to_csv(path + "submission_0208_1.csv", index=False)
# https://dacon.io/competitions/official/236214/mysubmission

# 점수 
# accuracy_score : 0.9361333871805227
# loss 0.17946083843708038
# f1 score 0.924449262145721


# BaggingClassifier 의 정답률: 0.863324294697975

# acc: [0.8655244  0.86012461 0.87237799 0.86677051 0.86644511 0.86644511
#  0.86551044 0.86281026 0.86447191 0.8730917 ]
#  평균 acc: 0.8664

# acc: [0.86417445 0.87154725 0.86760125 0.86656282 0.86800291 0.86665282
#  0.86364108 0.86966455 0.86976841 0.86291411]
#  평균 acc: 0.8671

# acc: [0.86386293 0.86801661 0.86531672 0.86666667 0.86436805 0.86737979
#  0.86509503 0.86395264 0.8677952  0.8640565 ]
#  평균 acc: 0.8657
# cross_val_precit acc: 0.818208042462355




