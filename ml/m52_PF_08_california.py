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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor,\
    VotingRegressor, StackingRegressor
from sklearn.preprocessing import LabelEncoder
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.decomposition import  PCA

plt.rcParams['font.family'] = 'Malgun Gothic'

import warnings

warnings.filterwarnings('ignore')


plt.rcParams['font.family'],"Malgun Gothic"
plt.rcParams['axes.unicode_minus']=False

#1 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_poly = scaler.fit_transform(x_poly)

xp_train, xp_test, yp_train, yp_test = train_test_split(
    x_poly, y, random_state=777, train_size=0.8,
)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
)

# 라벨 인코딩. StratifiedKFold 할때만 필요
label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)
# 라벨 인코딩. StratifiedKFold 할때만 필요

# pca = PCA(n_components=7)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#2. model
model = LinearRegression()
model2 = LinearRegression()
#3. 훈련
model.fit(x_train,y_train)
model2.fit(xp_train,yp_train)
#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score : ', model.score(x_test,y_test))

yp_predict = model2.predict(xp_test)
p_score = model2.score(xp_test, yp_test)
print('model2.score : ', p_score)

# model.score: 0.8534372129491651
# T r2: 0.858944147480187
# F r2: 0.8442317659339414

# VS r2: 0.8408760327999003

# model.score :  0.8411648807974154
# 스태킹 R2 : 0.8411648807974154

# model.score :  0.5827784121245208
# model2.score :  0.609106273785204