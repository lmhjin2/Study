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
import warnings

warnings.filterwarnings('ignore')

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

def objective(trial):
    # StratifiedKFold 설정
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 하이퍼파라미터 제안
    param = {
        'objective': 'binary',
        'metric': 'rmse',
        # 'verbosity': -1,
        # 'boosting_type': 'gbdt',
        # 'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
        # 'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0), 
        # 'learning_rate' : trial.suggest_float('learning_rate', 0.0049, 0.0051), # 5e-3
        'learning_rate' : trial.suggest_float('learning_rate', 1e-4, 1e-1), # 5e-3
        'max_depth' : trial.suggest_int('max_depth', 1, 50),
        'subsample' : trial.suggest_float('subsample',0.7, 1.0),
        'max_bin' : trial.suggest_int('max_bin', 200, 300),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        # 'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        # 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        # 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    
    rmses = []
    for train_idx, valid_idx in kfold.split(train_x, train_y):
        x_train_fold, x_valid_fold = train_x[train_idx], train_x[valid_idx]
        y_train_fold, y_valid_fold = train_y[train_idx], train_y[valid_idx]
        
        model = LGBMRegressor(**param, seed=9)
        model.fit(x_train_fold, y_train_fold, eval_set=[(x_valid_fold, y_valid_fold)])
        
        preds = model.predict(x_valid_fold)
        rmse = np.sqrt(mean_squared_error(y_valid_fold, preds))
        rmses.append(rmse)
    
    # 최소 RMSE 반환
    min_rmse = np.min(rmses)
    return min_rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=3000) # 100회의 시도로 최적화

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print(f'Best RMSE Score: {study.best_trial.value}')

trial = study.best_trial
best_params = study.best_trial.params

best_model = LGBMRegressor(**best_params, seed=9)
best_model.fit(train_x, train_y)

optuna.visualization.plot_param_importances(study)      # 파라미터 중요도 확인 그래프
optuna.visualization.plot_optimization_history(study)   # 최적화 과정 시각화

bpred = best_model.predict(test_x)

submission = pd.read_csv('d:/data/income/sample_submission.csv')
submission['Income'] = bpred
submission.to_csv('c:/Study/dacon/income/output/0326_opt.csv', index=False)

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))





'''
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 42 )

parameters = [{'learning_rate' : [0.00494997],  # 0.00495  / 0.00494992
               'max_depth' : [None],
               'gamma' : [1],
               'subsample' : [1],
               'max_bin' : [100],
               'colsample_bytree' : [0.5],
               'seed' : [9]
               
               }]
# best_rmse :  588.2896713143929 seed: 9 lr : 0.00495
# best_rmse :  588.2896767495979 seed: 9 lr : 0.00494992
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

submission.to_csv('c:/Study/dacon/income/output/0319_lgbm.csv', index=False)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_) 
print('best_rmse : ', best_rmse)

'''