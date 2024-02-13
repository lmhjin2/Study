import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, shuffle=True, random_state= 0, stratify=y)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

n_splits =  5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'RF__n_estimators':[100,200,300], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split': [2,3,5,10]},
    {'RF__min_samples_split': [2,3,5,10]}
    ]

#2 모델
pipe = Pipeline([('MaxAbs', MaxAbsScaler()),('PCA',PCA()), ('RF', RandomForestClassifier())])
model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.9298245614035088

y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test,y_predict))
    # accuracy_score: 0.9298245614035088
print('걸린시간:', np.round(end_time - strat_time, 2), '초')

# BaggingClassifier 의 정답률: 0.9651162790697675

# iter: 2
# n_candidates: 10
# n_resources: 108
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 200}
# best_score : 0.9714285714285713
# model.score : 0.9320175438596491
# accuracy_score: 0.9320175438596491
# 최적 튠 ACC: 0.9320175438596491
# 걸린시간: 45.55 초

# pipeline
# model.score : 0.9298245614035088
# accuracy_score: 0.9298245614035088
# 걸린시간: 0.05 초

# Pipeline
# model.score : 0.9298245614035088
# accuracy_score: 0.9298245614035088
# 걸린시간: 30.74 초