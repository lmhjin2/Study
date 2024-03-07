import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_squared_log_error
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
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

#2 model
xgb = XGBRegressor()
# xgb = LogisticRegression()
model = BaggingRegressor(xgb,
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=777,
                          bootstrap=True, # 기본값, 데이터 중복 허용. (샘플링)
                        #   bootstrap=False,    # 중복 허용 안함
                          
                          )

#3 compile train
model.fit(x_train, y_train)
#4 predict, test
results = model.score(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('model.score:', results)
print('r2:', results)

# best_score : 0.4511509586605877
# T r2: 0.3422603227510965
# F r2: 0.12475664605508174
