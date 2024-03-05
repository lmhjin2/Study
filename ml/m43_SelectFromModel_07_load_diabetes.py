import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_squared_log_error
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression, ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators
from xgboost import XGBRegressor
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings('ignore')


plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape)        # (442, 10)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 1)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# pca = PCA(n_components=8)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)
# parameters = {
#     'n_estimators' : [100,200,300,400,500],
#     'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1],
#     'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'gamma' : [0, 1, 2],
#     'lambda' : [0, 0.1, 0.01],
#     'alpha' : [0, 0.1, 0.01]
# }
parameters = {'n_estimators':1000,
              'learning_rate': 0.01,
              'max_depth':3,
              'gamma':0,
              'min_child_weight':0,
              'subsample':0.4,  # dropout개념과 비슷
              'colsample_bytree':0.8,
              'colsample_bylevel':0.7,
              'colsample_bynode':1,
              'reg_alpha': 0,
              'reg_lamda': 1,
              'random_state': 3377,
              'verbose' :0
              }
#2 model
model = XGBRegressor(random_state=0)
model.set_params(early_stopping_rounds = 10, **parameters)

#3 compile train
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose= 0 )
#4 predict, test
results = model.score(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('model.score:', results)
print('r2:', results)

# 오름
# 최적의 파라미터 :  {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.1, 'lambda': 0, 'gamma': 1, 'alpha': 0.1}
# best_score : 0.4511509586605877
# 걸린시간: 2.53 초

####################################################################################

# model.score: 0.38900125504969674
# r2: 0.38900125504969674

# 4: 0.007262748940967012

####################################################################################

print("="*50)
thresholds = np.sort(model.feature_importances_)
# print(thresholds)
from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(i, "\t변형된 x_train :", select_x_train.shape, "변형된 x_test :", select_x_test.shape)
    # i 에는 thresholds 밑에 숫자가 하나씩 나오는데 그거 이상의 숫자를 가진 컬럼은 다 살아남고 / 그거 미만인 컬럼은 사라지는 구조
    select_model = XGBRegressor()
    select_model.set_params(
        early_stopping_rounds=10, **parameters,
    )
    
    select_model.fit(select_x_train, y_train,
                     eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                     verbose=0,)
    select_y_predict = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)
    print("Trech=%.3f, n=%d, r2 %.2f%%" % (i,select_x_train.shape[1], score*100))

# model.score: 0.38900125504969674
# r2: 0.38900125504969674
# Trech=0.087, n=6, r2 39.63%

