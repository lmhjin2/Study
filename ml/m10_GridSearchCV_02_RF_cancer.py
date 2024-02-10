import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
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

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, shuffle=True, random_state= 0, stratify=y)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12], 'min_samples_leaf':[3,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split': [2,3,5,10]},
    {'min_samples_split': [2,3,5,10]},
    {'n_jobs':[-1,2,4], 'min_samples_split': [2,3,5,10]}
    ]

#2 모델

model = GridSearchCV(RandomForestClassifier(), parameters, cv = kfold,
                    # verbose=1, 
                    refit = True, 
                    # n_jobs=-1     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
                    )
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print("최적의 매개변수 : ", model.best_estimator_)
    # 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=4)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
    # 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': 4}
print('best_score :', model.best_score_)
    # best_score : 0.9742424242424242
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.9298245614035088

y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test,y_predict))
    # accuracy_score: 0.9298245614035088
y_pred_best = model.best_estimator_.predict(x_test)
            # 최적의 매개변수.predict(x_test)
print('최적 튠 ACC:', accuracy_score(y_test,y_pred_best))
    # 최적 튠 ACC: 0.9298245614035088

print('걸린시간:', np.round(end_time - strat_time, 2), '초')

# BaggingClassifier 의 정답률: 0.9651162790697675

# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=4)
# 최적의 파라미터 :  {'min_samples_split': 3, 'n_jobs': 4}
# best_score : 0.9742424242424242
# model.score : 0.9298245614035088
# accuracy_score: 0.9298245614035088
# 최적 튠 ACC: 0.9298245614035088
# 걸린시간: 45.73 초
