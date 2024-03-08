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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor,StackingRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PolynomialFeatures
from sklearn.utils import all_estimators
from xgboost import XGBRegressor
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
plt.rcParams['font.family'] = 'Malgun Gothic'
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
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

# pca = PCA(n_components=8)
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

# best_score : 0.4511509586605877
# T r2: 0.3422603227510965
# F r2: 0.12475664605508174

# VS r2: 0.24160631151890655

# model.score :  0.3119202938666691
# 스태킹 R2 : 0.3119202938666691

# model.score :  0.5628181668953596
# model2.score :  0.444149143357622