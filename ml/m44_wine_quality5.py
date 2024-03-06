# [실습]
# 1. 아웃라이어 확인
# 2. 아웃라이어 처리
# 3. 44_1이든 44_2든 수정해서 맹그러

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

path = 'c:/_data/dacon/wine/'

### 맹그러 ###

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
# y = y-3
# print(y)
y = y.copy()
# (5497,13)
##############################################################
# [실습] y의 클래스를 7개에서 5~3개로 줄여서 성능 비교
##############################################################
# y = y.copy()    알아서 참고
## 힌트 : for문 돌리면 됨

for i, v in enumerate(y):
    if v<=4:
        y[i] = 0
    elif v==5:
        y[i] = 1
    elif v==6:
        y[i] = 2
    elif v==7:
        y[i] = 3
    elif v==8:
        y[i] = 4        
    else:
        y[i] = 5

print(y.value_counts().sort_index())

def remove_outlier(dataset:pd.DataFrame):
    for label in dataset:
        data = dataset[label]
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3-q1
        upbound    = q3 + iqr*1.5
        underbound = q1 - iqr*1.5
        dataset.loc[dataset[label] < underbound, label] = underbound
        dataset.loc[dataset[label] > upbound, label] = upbound
        
    return dataset

x = remove_outlier(x)
x = x.astype(np.float32)
y = y.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, train_size=0.8, random_state= 3 )
# print(x_train.shape, x_test.shape)  
#2
model = RandomForestClassifier()

#3 compile train
model.fit(x_train, y_train)

#4 predict, test
results = model.score(x_test, y_test)
print("최종점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc: ", acc)

f1 = f1_score(y_test, y_predict, average='macro')
print("f1:", f1)
