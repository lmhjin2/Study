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
x = datasets.data
y = datasets['target']
# x,y = load_iris(return_X_y=True)
# print(x.shape, y.shape)       # (150, 4) (150,)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify= y, train_size = 0.8, random_state = 0 )

# 이름 직접 바꾸기.
class CustomXGBClassifier(XGBClassifier):
    def __repr__(self):           
        return "XGBClassifier(random_state= 0)"
## 이름 직접 바꾸기2.
# class CustomXGBClassifier(XGBClassifier):
#     def __str__(self):           
#         return "XGBClassifier(random_state= 0)"
#2
models = [DecisionTreeClassifier(random_state= 0), RandomForestClassifier(random_state= 0),
          GradientBoostingClassifier(random_state= 0), CustomXGBClassifier(random_state= 0)]

np.set_printoptions(suppress=True)

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

import matplotlib.pyplot as plt

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.figure(figsize=[14,7])
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)
#     plt.title(model)

# plot_feature_importances_dataset(model)
# plt.show()
    
from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()

