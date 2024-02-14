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
## numpy
x = np.delete(x, 0, axis=1)
# print(x.shape, y.shape)       # (150, 4) (150,)
# print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify= y, train_size = 0.8, random_state = 0 )

## pandas
# pd.DataFrame
# 컬럼명 : datasets.feature_names 안에 있음

# aaa = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# df = pd.DataFrame(data = x, columns=aaa, index=y)
# # print(load_iris().feature_names)
# # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# df = df.drop('sepal length (cm)', axis=1)
df = pd.DataFrame(data=load_iris().data, columns = load_iris().feature_names)
df = df.drop('sepal length (cm)', axis=1)
# print(df)
X = df #.drop(columns=['sepal length (cm)'])
Y = load_iris().target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

#2
models = [DecisionTreeClassifier(random_state= 0), RandomForestClassifier(random_state= 0),
          GradientBoostingClassifier(random_state= 0), XGBClassifier(random_state= 0)]

# np.set_printoptions(suppress=True)

# for model in models:
#     try:
#         model.fit(x_train, y_train)
#         results = model.score(x_test, y_test)
#         y_predict = model.predict(x_test)
#         print(type(model).__name__, "accuracy score", accuracy_score(y_predict, y_test))
#         print(type(model).__name__, "model.score", results)
#         print(type(model).__name__, ":", model.feature_importances_, end='\n\n')
#         ## type(model).__name__ == 모델 이름만 뽑기
#         # end = '\n\n' == print('\n') 한줄 추가
#     except Exception as e:
#         print("에러:", e)
#         continue

for model in models:
    try:
        model.fit(X_train, Y_train)
        results = model.score(X_test, Y_test)
        y_predict = model.predict(X_test)
        print(type(model).__name__, "accuracy score", accuracy_score(y_predict, Y_test))
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_, end='\n\n')
        ## type(model).__name__ == 모델 이름만 뽑기
        # end = '\n\n' == print('\n') 한줄 추가
    except Exception as e:
        print("에러:", e)
        continue
    
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


# DecisionTreeClassifier accuracy score 0.9666666666666667
# DecisionTreeClassifier model.score 0.9666666666666667
# DecisionTreeClassifier : [0.01666667 0.52797619 0.45535714]

# RandomForestClassifier accuracy score 1.0 ==================================== 오름
# RandomForestClassifier model.score 1.0
# RandomForestClassifier : [0.11184392 0.47236891 0.41578716]

# GradientBoostingClassifier accuracy score 0.9666666666666667
# GradientBoostingClassifier model.score 0.9666666666666667
# GradientBoostingClassifier : [0.01138601 0.28944383 0.69917016]

# XGBClassifier accuracy score 0.9333333333333333
# XGBClassifier model.score 0.9333333333333333
# XGBClassifier : [0.03150374 0.5808004  0.38769582]