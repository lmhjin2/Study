import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron, LogisticRegression
                                            # regression이지만 의외로 분류
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#1
datasets = load_iris()
x,y = load_iris(return_X_y=True)
# print(x.shape, y.shape)       # (150, 4) (150,)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify= y, train_size = 0.8, random_state = 0 )

#2
models = [DecisionTreeClassifier(random_state= 0), RandomForestClassifier(random_state= 0),
          GradientBoostingClassifier(random_state= 0), XGBClassifier(random_state= 0)]

# np.set_printoptions(suppress=True)

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "accuracy score", accuracy_score(y_predict, y_test))
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_, end='\n\n')
        ## type(model).__name__ == 모델 이름만 뽑기
        # end = '\n\n' == print('\n') 한줄 추가
    except Exception as e:
        print("에러:", e)
        continue

# np.set_printoptions(suppress=False)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

models = [DecisionTreeRegressor(random_state= 0), RandomForestRegressor(random_state= 0),
          GradientBoostingRegressor(random_state= 0), XGBRegressor(random_state= 0)]

## 이름 직접 바꾸기.
# class CustomXGBClassifier(XGBClassifier):
#     def __repr__(self):           
#         return "XGBClassifier(random_state= 0)"

#     def __str__(self): 이거 써도됨






# DecisionTreeClassifier(random_state=0) model.score : 0.9666666666666667
# DecisionTreeClassifier(random_state=0) accuracy_score : 0.9666666666666667
# DecisionTreeClassifier(random_state=0) : [0.01666667 0.         0.02797619 0.95535714]
# 중요도 : 4번째 컬럼 >>>>> 3번째 컬럼 > 첫번째 컬럼 >> 두번째 컬럼

# RandomForestClassifier(random_state=0) model.score : 0.9333333333333333
# RandomForestClassifier(random_state=0) accuracy_score : 0.9333333333333333
# RandomForestClassifier(random_state=0) : [0.09603851 0.03548464 0.38665429 0.48182255]
# 중요도 : 4번째 컬럼 >>>>> 3번째 컬럼 > 첫번째 컬럼 >> 두번째 컬럼

# GradientBoostingClassifier(random_state=0) model.score : 0.9666666666666667
# GradientBoostingClassifier(random_state=0) accuracy_score : 0.9666666666666667
# GradientBoostingClassifier(random_state=0) : [0.0081988  0.01273213 0.27045856 0.70861051]
# 중요도 : 4번째 컬럼 >>>>> 3번째 컬럼 > 2번째 컬럼 >> 첫번째 컬럼

# XGBClassifier(random_state=0) model.score : 0.9333333333333333
# XGBClassifier(random_state=0) accuracy_score : 0.9333333333333333
# XGBClassifier(random_state=0) : [0.0196958  0.01798598 0.59199977 0.3703184 ]
# 중요도 : 3번째 컬럼 >>>>> 4번째 컬럼 > 첫번째 컬럼 >> 2번째 컬럼

