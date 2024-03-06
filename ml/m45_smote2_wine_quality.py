# acc, f1 둘다 확인
# acc, f1 둘다 확인
# acc, f1 둘다 확인
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

path = 'c:/_data/dacon/wine/'

### 맹그러 ###

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함

train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
# y = y-3
y = y.copy()
# print(pd.value_counts(y))
# quality
# 3    2416
# 2    1788
# 4     924
# 1     186
# 5     152
# 0      26
# 6       5
for i, v in enumerate(y):
    if v==3:
        y[i] = 0
    elif v==4:
        y[i] = 1
    elif v==5:
        y[i] = 2
    elif v==6:
        y[i] = 3
    elif v==7:
        y[i] = 4
    elif v==8:
        y[i] = 5       
    else:
        y[i] = 6
print(y.value_counts().sort_index())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state= 3 )

print(x_train.shape, y_train.shape)

smote = SMOTE(k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(x_train.shape, y_train.shape)

#2
model = XGBClassifier()

#3 compile train
model.fit(x_train, y_train)

#4 predict, test
results = model.score(x_test, y_test)
print("최종점수 : ", results)

y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("acc: ", acc)

f1 = f1_score(y_test, y_predict, average='macro')
print("f1: ", f1)

# (4397, 12) (4397,)
# acc:  0.6436363636363637
# f1:  0.38378834035885745
# SMOTE
# (13587, 12) (13587,) 떨어짐
# acc:  0.6118181818181818
# f1:  0.37108644502341004
# All Label
# (13587, 12) (13587,)
# 최종점수 :  0.62
# f1:  0.3793877173077398
# 3 Label
# (7764, 12) (7764,)
# 최종점수 :  0.6618181818181819
# f1:  0.6235723482027618