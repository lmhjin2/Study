import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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
y = LabelEncoder().fit_transform(y)
print(np.unique(y,return_counts=True))

from sklearn.preprocessing import OneHotEncoder
y_ohe = y.reshape(-1, 1)
enc = OneHotEncoder(sparse=False).fit(y_ohe)
y_ohe = enc.transform(y_ohe)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, train_size=0.8, random_state= 3 )
print(x_train.shape, x_test.shape)

scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators':1000,
              'learning_rate': 0.015,
              'max_depth':7,
              'gamma':0,
              'min_child_weight': 0 ,
              'subsample':0.4,  # dropout개념과 비슷
              'colsample_bytree': 0.8 ,
              'colsample_bylevel': 0.7 ,
              'colsample_bynode': 1 ,
              'reg_alpha': 0 ,
            #   'reg_lamda': 1 ,
              'random_state': 3377 ,
            #   'verbose' : 0
              }    # acc:  0.6836363636363636363636    

#2
model = XGBClassifier()
model.set_params(early_stopping_rounds = 10, **parameters)

#3 compile train
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose= 0 )

#4 predict, test
results = model.score(x_test, y_test)
print("최종점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc: ", acc)



