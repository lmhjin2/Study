import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

#1
datasets = load_iris()
x = datasets.data
y = datasets['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, train_size=0.8, random_state=0)

# 이름 직접 바꾸기.
class CustomXGBClassifier(XGBClassifier):
    def __repr__(self):           
        return "XGBClassifier(random_state= 0)"
## 이름 직접 바꾸기2.
# class CustomXGBClassifier(XGBClassifier):
#     def __str__(self):
#         return "XGBClassifier(random_state= 0)"
model1 = DecisionTreeClassifier(random_state= 0)
model2 = RandomForestClassifier(random_state= 0)
model3 = GradientBoostingClassifier(random_state= 0)
model4 = XGBClassifier(random_state= 0)
#2
models = [model1, model2, model3, model4]

plt.figure(figsize=[16, 12])

for i, model in enumerate(models, 1):
    try:
        ax = plt.subplot(2, 2, i)
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "accuracy score", accuracy_score(y_predict, y_test))
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_)
        n_features = datasets.data.shape[1]
        ax.barh(np.arange(n_features), model.feature_importances_, align='center')
        ax.set_yticks(np.arange(n_features))
        ax.set_yticklabels(datasets.feature_names)
        ax.set_xlabel("Feature Importances")
        ax.set_ylabel("Features")
        ax.set_title(type(model).__name__)
    except Exception as e:
        print("에러:", e)
        continue

plt.tight_layout()
plt.show()
