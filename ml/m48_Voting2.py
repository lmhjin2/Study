import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')
np.set_printoptions()

#1. data
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
    stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. model
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = VotingClassifier(
    estimators= [('LR',lr),('RF',rf),('XGB', xgb)],
    # voting='soft',
    voting='hard',  # 기본값
)

#3. train
model.fit(x_train, y_train)

#4. test, predict
results = model.score(x_test, y_test)
print("최종점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc: ", acc)

####################################################################################
# 19개 컬럼이 삭제되었을 때 Score:  0.956140350877193
# 19: 0.00877192982456143

# 20개 컬럼이 삭제되었을 때 Score:  0.956140350877193
# 20: 0.00877192982456143

# T acc:  0.9649122807017544
# F acc:  0.956140350877193

# VS acc:  0.9824561403508771
# VH acc:  0.9824561403508771
model_class = [xgb,rf,lr]

for model2 in model_class:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print("{0} 정확도 : {1:.4f}".format(class_name, score2))


# 최종점수 :  0.9824561403508771
# acc:  0.9824561403508771
# XGBClassifier 정확도 : 0.9912
# RandomForestClassifier 정확도 : 0.9825
# LogisticRegression 정확도 : 0.9737
