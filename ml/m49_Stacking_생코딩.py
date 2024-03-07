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
from catboost import CatBoostClassifier
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

models = [xgb,rf,lr]
for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # print(y_pred.shape) # (114,)
    score = accuracy_score(y_test, y_pred)
    class_name = model.__class__.__name__
    print("{0} Score : {1:.4f}".format(class_name, score))

x_train_stack = []
x_test_stack = []

for model in models:
    x_train_stack.append(model.predict(x_train))
    x_test_stack.append(model.predict(x_test))

# x_train_stack = np.array(x_train_stack)
# x_test_stack = np.array(x_test_stack)

# print(x_train_stack.shape, x_test_stack.shape)   # (3, 455) (3, 114)
x_train_stack = np.array(x_train_stack).T
x_test_stack = np.array(x_test_stack).T
# print(x_train_stack.shape, x_test_stack.shape)   # (455, 3) (114, 3)

model2 = CatBoostClassifier(verbose=0)
model2.fit(x_train_stack,y_train)
y_pred2 = model2.predict(x_test_stack)
score2 = accuracy_score(y_test, y_pred2)
print("스태킹 결과 :", score2)

# XGBClassifier Score : 0.9912
# RandomForestClassifier Score : 0.9649
# LogisticRegression Score : 0.9737
# 스태킹 결과 : 0.9824561403508771
# 위가 내거
#################################################################################
# 아래가 쌤꺼
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

models = [xgb, rf, lr]
li = []
li2 = []

for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    li.append(y_pred)
    li2.append(y_predict)
    class_name = model.__class__.__name__
    print("{0} 정확도 : {1:.4f}".format(class_name, score))

new_x_train = np.array(li).T
new_x_test = np.array(li2).T

model2 = CatBoostClassifier(verbose=0)
model2.fit(new_x_train, y_train)
y_pred3 = model2.predict(new_x_test)
score2 = accuracy_score(y_test, y_pred3)
print("스태킹 결과 : ", score2)

# XGBClassifier 정확도 : 0.9912
# RandomForestClassifier 정확도 : 0.9737
# LogisticRegression 정확도 : 0.9737
# 스태킹 결과 :  0.9824561403508771