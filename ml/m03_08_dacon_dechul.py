# https://dacon.io/competitions/official/236214/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time as tm
from sklearn.svm import LinearSVC, LinearSVR

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

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2
# model = LinearSVC()

# #3
# # import datetime

# # date = datetime.datetime.now()
# # date = date.strftime("%m%d_%H%M")   # 월일_시분

# # path1 = "c:/_data/_save/MCP/k28/11/"
# # filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
# # filepath = "".join([path1, 'k28_', date, '_1_', filename])
# start_time = tm.time()

# model.fit(x_train, y_train)

# end_time = tm.time()
# run_time = round(end_time - start_time, 2)

# #4
# acc = model.score(x_test, y_test)
# y_submit = model.predict(test_csv)
# y_predict = model.predict(x_test)

# y_submit = le_grade.inverse_transform(y_submit)

# submission_csv['대출등급'] = y_submit
# submission_csv.to_csv(path + "submission_0205_1.csv", index=False)
# # https://dacon.io/competitions/official/236214/mysubmission

# accuracy = accuracy_score(y_test, y_predict)
# f1 = f1_score(y_test, y_predict, average = 'macro') # [None, 'micro', 'macro', 'weighted'] 중에 하나

# print('acc', acc)
# print('accuracy_score :', accuracy)
# print('f1 score', f1)
# print('run time', run_time)

# 점수 0.91392
# accuracy_score : 0.9356141464258928
# loss 0.18289394676685333
# f1 score 0.9230744496445683

# 점수 
# accuracy_score : 0.9361333871805227
# loss 0.17946083843708038
# f1 score 0.924449262145721

from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model_P = Perceptron()
model_L = LogisticRegression()
model_K = KNeighborsClassifier()
model_D = DecisionTreeClassifier()
model_R = RandomForestClassifier()

models = [model_P, model_L,  model_K, model_D, model_R]

for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(model, "acc", results)

model_P = Perceptron()
model_L = LinearRegression()
model_K = KNeighborsRegressor()
model_D = DecisionTreeRegressor()
model_R = RandomForestRegressor()

# Perceptron() acc 0.38250735591069057
# LogisticRegression() acc 0.37841112329083254
# KNeighborsClassifier() acc 0.45341256562626203
# DecisionTreeClassifier() acc 0.833266024346622
# RandomForestClassifier() acc 0.8047077828419777
