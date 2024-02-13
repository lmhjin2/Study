#### 데이터셋 마다 최상의 파라미터를 알고 있다.!!
# GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
# 그래도 적용해서 13개 맹그러

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
                                            
#1
datasets = load_iris()
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify= y, train_size = 0.8, random_state = 0 )

print(np.min(x_train), np.max(x_train)) # 0.1 / 7.9
print(np.min(x_test), np.max(x_test)) # 0.1 / 7.2

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2
# model = RandomForestClassifier()
model = make_pipeline(MaxAbsScaler(), RandomForestClassifier())
# MaxAbsScaler, RobustScaler 가 더 잘나옴
#3
model.fit(x_train, y_train)

#4
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)

# print(y_predict)
acc = accuracy_score(y_predict, y_test)

print('model.score :', results)
print("accuracy_score :", acc)

## model.score 
# 분류모델에선 acc, 회귀에서는 r2
# 회귀에서는 LinearSVR 씀                                                                                       

# model = RandomForestClassifier()
# model.score : 0.9666666666666667
# accuracy_score : 0.9666666666666667


