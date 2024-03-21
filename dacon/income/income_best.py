import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import optuna

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

parameters = [{'learning_rate' : [0.00494997],  # 0.00495  / 0.00494992
               'max_depth' : [None],
            #    'gamma' : [1],
               'subsample' : [1],
               'max_bin' : [100],
               'colsample_bytree' : [0.5],
               'seed' : [9]
               }]
# best_rmse :  587.0828680513683

#2
model = GridSearchCV(LGBMRegressor(n_estimators = 1000 , 
                    #   learning_rate = 0.00495 , 
                    #   max_depth = None ,
                    # #   min_child_weight= 35.723980094661194 ,
                    #   gamma = 1 ,  
                    #   subsample = 1 ,
                    #   max_bin = 100 ,
                    #   colsample_bytree= 0.5 ,
                    #   objective= 'binary:logistic' ,
                    #   nthread= 1 ,
                      # scale_pos_weight= 1 , # 양수데이터가 적을때 양수 데이터 중요도 올리기. 10 = 10배
                      ), parameters, cv=kfold, refit=True, n_jobs=-1 )
#3
model.fit(x_train, y_train) 

#4
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
best_y_predict = model.best_estimator_.predict(x_test)
# mse
mse = mean_squared_error(y_test, y_predict)
best_mse = mean_squared_error(y_test, best_y_predict)
# rmse
rmse = np.sqrt(mse)
best_rmse = np.sqrt(best_mse)
# submission
preds = model.predict(test_x)
best_pred = model.best_estimator_.predict(test_x)

submission = pd.read_csv('d:/data/income/sample_submission.csv')
submission['Income'] = preds
# print(submission)

submission.to_csv('c:/Study/dacon/income/output/best.csv', index=False)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_) 
print('best_rmse : ', best_rmse)

