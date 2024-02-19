import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import all_estimators
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(x.shape)    # (581012, 54)
scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(np.unique(y, return_counts=True))  # (581012,)

# ohe = OneHotEncoder(sparse=False, categories='auto')
# y = y.reshape(-1,1)
# y_ohe = ohe.fit_transform(y)
# print(x.shape)

y = y-1  ## 야매로 굴러가게한거임.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)
n_splits = 3
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)
parameters = {
    'n_estimators' : [100,200,300,400,500],
    'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1],
    'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'gamma' : [0, 1, 2],
    'lambda' : [0, 0.1, 0.01],
    'alpha' : [0, 0.1, 0.01]
}
#2 model
xgb = XGBClassifier(random_state=0)
model = RandomizedSearchCV(xgb, parameters, cv = kfold, refit=True, n_jobs= 22 )

#3 compile train
import time as tm
start_time = tm.time()
model.fit(x_train,y_train)
end_time = tm.time()
#4 predict, test
# y_ohe = ohe.inverse_transform(y_ohe)
results = model.score(x_test,y_test)
y_predict = model.predict(x_test)
y_pred_best = model.best_estimator_.predict(x_test)
acc = accuracy_score(y_test, y_predict)
scores = cross_val_score(model, x_test, y_test, cv=kfold)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
print('best_score :', model.best_score_)
print('최적 튠 ACC:', accuracy_score(y_test,y_pred_best))
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
print('model.score:', results)
print('acc:', acc)
print('걸린시간:', np.round(end_time - start_time, 2), '초')

# model.score : 0.8850976308701153
# 오름
# 최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 9, 'learning_rate': 0.3, 'lambda': 0.01, 'gamma': 0, 'alpha': 0.1}
# best_score : 0.9590369417328016
# 최적 튠 ACC: 0.9620233556792854
# acc: [0.89492707 0.92414268 0.89462588 0.90959552 0.90843373]
#  평균 acc: 0.9063
# model.score: 0.9620233556792854
# acc: 0.9620233556792854
# 걸린시간: 465.52 초