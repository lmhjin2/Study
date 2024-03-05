import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,mean_absolute_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import  PCA

import warnings

warnings.filterwarnings('ignore')


plt.rcParams['font.family'],"Malgun Gothic"
plt.rcParams['axes.unicode_minus']=False

#1 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# 라벨 인코딩. StratifiedKFold 할때만 필요
label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)
# 라벨 인코딩. StratifiedKFold 할때만 필요
# print(x.shape, y.shape)   # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# pca = PCA(n_components=7)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)
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


# 최적의 파라미터 :  {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1, 'lambda': 0, 'gamma': 1, 'alpha': 0.01}
# 최적 튠 R2: 0.8534372129491651
# model.score: 0.8534372129491651
# r2: 0.8534372129491651
# 걸린시간: 7.85 초


####################################################################################

# model.score: 0.7839574065512667
# r2: 0.7839574065512667
# 전부 - 
# {0: 0.0, 1: -0.0018526165048676768, 2: -0.007118538018523468, 
# 3: -0.006655838178142193, 4: -0.09853630068013286, 5: -0.16751677706960977, 
# 6: -0.23231570755290676, 7: -0.3252800818780236}

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


