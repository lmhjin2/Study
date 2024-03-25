import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression

#1
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

train = pd.read_csv('d:/data/income/train.csv')
test = pd.read_csv('d:/data/income/test.csv')

train_x = train.drop(columns=['ID', 'Income'])
train_y = train['Income']
test_x = test.drop(columns=['ID'])

encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    
    # train과 test 데이터셋에서 해당 열의 모든 값을 문자열로 변환
    train_x[i] = train_x[i].astype(str)
    test_x[i] = test_x[i].astype(str)
    
    le.fit(train_x[i])
    train_x[i] = le.transform(train_x[i])
    
    # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
    for case in np.unique(test_x[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
    
    test_x[i] = le.transform(test_x[i])

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state= 42)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 42 )


#2
lgbm = LGBMRegressor(n_estimators = 1000 , 
                     learning_rate = 0.00494997 , 
                     max_depth = None ,
                     subsample = 1 ,
                     max_bin = 100 ,
                     colsample_bytree= 0.5 ,
                     seed=9)

lgbm1 = LGBMRegressor(n_estimators = 1000 , 
                     learning_rate = 0.00494997 , 
                     max_depth = None ,
                     subsample = 1 ,
                     max_bin = 100 ,
                     colsample_bytree= 0.5 ,
                     seed=9)

lgbm2 = LGBMRegressor(n_estimators = 1000 , 
                     learning_rate = 0.00494997 , 
                     max_depth = None ,
                     subsample = 1 ,
                     max_bin = 100 ,
                     colsample_bytree= 0.5 ,
                     seed=9)

lgbm3 = LGBMRegressor(n_estimators = 1000 , 
                     learning_rate = 0.00494997 , 
                     max_depth = None ,
                     subsample = 1 ,
                     max_bin = 100 ,
                     colsample_bytree= 0.5 ,
                     seed=9)

xgb = XGBRegressor(n_estimators=1000,
                   learning_rate=0.00494997,
                   max_depth = None,
                   gamma = 1,
                   subsample = 1,
                   max_bin = 100,
                   colsample_bytree = 0.5,
                   seed=9
                   )

cbt = CatBoostRegressor(n_estimators=1000,
                        learning_rate= 0.014795,
                        random_seed=9
                        )



base_models = [
    ('lgbm1', lgbm1),
    ('lgbm2', lgbm2),
    ('lgbm3', lgbm3)
    ]

meta_model = lgbm

stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=kfold
)
#3
stacking.fit(x_train, y_train) 

#4
results = stacking.score(x_test, y_test)
y_predict = stacking.predict(x_test)
# mse
mse = mean_squared_error(y_test, y_predict)
# rmse
rmse = np.sqrt(mse)
# submission
preds = stacking.predict(test_x)

submission = pd.read_csv('d:/data/income/sample_submission.csv')
submission['Income'] = preds
# print(submission)

submission.to_csv('c:/Study/dacon/income/output/0325_ens.csv', index=False)

print('RMSE : ', rmse)

