import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
#1. 데이터
x, y = load_linnerud(return_X_y=True)
# print(x.shape, y.shape)    # (20, 3), (20, 3)
# print(y)
# 최종값 : x : [2 110 43],
#          y : [138 33 68]

#234. 모델, 훈련, 평가
# model = LGBMRegressor() # 에러. column이 여러개면 안먹힌다
# model = XGBRegressor() # best
# model = CatBoostRegressor() # 에러
# model = MultiOutputRegressor(CatBoostRegressor(verbose=0))
model = CatBoostRegressor(verbose=0, loss_function='MultiRMSE')
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__,'스코어 : ',
      round(mean_absolute_error(y, y_pred),4))  
print(model.predict([[2,110,43]]))

# RandomForestRegressor 스코어 :  3.6788
# LinearRegression 스코어 :  7.4567
# Ridge 스코어 :  7.4569
# Lasso 스코어 :  7.4629
# XGBRegressor 스코어 :  0.0008

## feat.MultiOutputRegressor
# LGBMRegressor 스코어 :  8.91
# CatBoostRegressor 스코어 :  0.2154
