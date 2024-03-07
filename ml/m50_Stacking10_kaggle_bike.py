import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, \
    GridSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV, KFold
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, VotingRegressor,\
    StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')


# 데이터 불러오기
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']
# print(x.shape)        # (10886, 11)

# 훈련 및 테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# pca = PCA(n_components=9)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

#2. model
from catboost import CatBoostRegressor
xgb = XGBRegressor()
rf = RandomForestRegressor()
lr = LinearRegression()

model = StackingRegressor(
    estimators=[('LR',lr),('RF',rf), ('XGB',xgb)],
    final_estimator=CatBoostRegressor(verbose=0),
    n_jobs=-1,
    cv=5,
)
#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score : ', model.score(x_test,y_test))
print('스태킹 R2 :', r2_score(y_test, y_predict))


# XGBRegressor model.score 0.9943523513944535

# best_score : 0.3410296277912012

# T r2: 0.33566399389485146
# F r2: 0.3264614470592798

# VS r2: 0.33074903834805647

# model.score :  0.34291680028254035
# 스태킹 R2 : 0.34291680028254035