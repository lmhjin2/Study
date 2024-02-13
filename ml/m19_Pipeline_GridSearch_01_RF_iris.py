import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA       # 주성분 분해 == 차원 축소 (열, 컬럼, 피처, 특성, 차원)


#1
datasets = load_iris()
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify= y, train_size = 0.8, random_state = 0 )
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

parameters = [
    {'RF__n_estimators':[100,200,300], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split': [2,3,5,10]},
    {'RF__min_samples_split': [2,3,5,10]}
    ]
#2
pipe = Pipeline([('MinMax', MinMaxScaler()), ('RF', RandomForestClassifier(max_depth=12, min_samples_leaf=5, n_estimators=300))])
model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

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


