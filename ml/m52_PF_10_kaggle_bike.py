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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
import warnings

warnings.filterwarnings('ignore')

# 데이터 불러오기
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']
# print(x.shape)        # (10886, 11)
pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)

scaler = StandardScaler()
x = scaler.fit_transform(x)
x_poly = scaler.fit_transform(x_poly)

xp_train, xp_test, yp_train, yp_test = train_test_split(
    x_poly, y, random_state=777, train_size=0.8,
)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
)
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


# XGBRegressor model.score 0.9943523513944535

# best_score : 0.3410296277912012

# T r2: 0.33566399389485146
# F r2: 0.3264614470592798

# VS r2: 0.33074903834805647

# model.score :  0.34291680028254035
# 스태킹 R2 : 0.34291680028254035

# model.score :  0.2504177567774257
# model2.score :  0.2815375380342934