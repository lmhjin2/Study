# https://dacon.io/competitions/official/236336/mysubmission

import pandas as pd
import numpy as np
import os
import random

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdFingerprintGenerator  # MorganGenerator를 사용하기 위해 추가
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm
import optuna

CFG = {
    'NBITS': 2048,
    'SEED': 42,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(CFG['SEED'])  # Seed 고정

# MorganGenerator 생성
morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=CFG['NBITS'])

# SMILES 데이터를 분자 지문으로 변환
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = morgan_generator.GetFingerprint(mol)  # MorganGenerator 사용
        return np.array(fp)
    else:
        return np.zeros((CFG['NBITS'],))

# 학습 ChEMBL 데이터 로드
chembl_data = pd.read_csv('c:/data/dacon/medicine/train.csv')
# print(chembl_data.head())

# print(chembl_data.shape) # (1952, 15) 

train = chembl_data[['Smiles', 'pIC50']].copy()  # .copy()를 사용하여 경고 해결
train.loc[:, 'Fingerprint'] = train['Smiles'].apply(smiles_to_fingerprint)  # .loc를 사용하여 경고 해결
# print(train.head())

train_x = np.stack(train['Fingerprint'].values)
train_y = train['pIC50'].values

# 학습 및 검증 데이터 분리
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
def objective(trial):
    # KFold 설정
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 하이퍼파라미터 제안
    param = {
        'objective': 'regression', # 기본값 regression
        # 'boosting_type' : ['gbdt', 'dart', 'goss', 'rf'],
        'metric': 'rmse',
        # 'n_estimators' : '100', # 기본 100
        'learning_rate' : trial.suggest_float('learning_rate', 0.1, 0.2),
        'min_child_weight' : trial.suggest_float('min_child_weight', 0.001, 0.1), # 기본 0.001
        'max_depth' : trial.suggest_int('max_depth', 16, 255), # 기본 -1 / 제한 x
        'subsample' : trial.suggest_float('subsample', 0.7, 1.0), # 기본 1.0
        'max_bin' : trial.suggest_int('max_bin', 25, 500), # 기본 255
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.7, 1.0), # 기본 1
        'num_leaves': trial.suggest_int('num_leaves', 8, 64), # 기본 31
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 30), # 기본 20
        # 'importance_type' : ['split', 'gain'],
        # 'boost_from_average' : True,
        # 'early_stopping_rounds' : None,
        # 'feature_fraction' : trial.suggest_float('feature_fraction', 0.9, 1.0),
        # 'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.9, 1.0),
        
    }

    rmses = []
    for train_idx, valid_idx in kfold.split(train_x, train_y):
        x_train_fold, x_valid_fold = train_x[train_idx], train_x[valid_idx]
        y_train_fold, y_valid_fold = train_y[train_idx], train_y[valid_idx]
        
        model = lgbm.LGBMRegressor(**param, seed=42)
        model.fit(x_train_fold, y_train_fold, eval_set=[(x_valid_fold, y_valid_fold)])
        
        preds = model.predict(x_valid_fold)
        rmse = np.sqrt(mean_squared_error(y_valid_fold, preds))
        rmses.append(rmse)
    
    # 최소 RMSE 반환
    min_rmse = np.min(rmses)
    return min_rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 100) # 10회의 시도로 최적화

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print(f'Best RMSE Score: {study.best_trial.value}')

trial = study.best_trial
best_params = study.best_trial.params

best_model = lgbm.LGBMRegressor(**best_params, seed=42)
best_model.fit(train_x, train_y)

import optuna.visualization as vis

# 파라미터 중요도 확인 그래프
fig_param_importances = vis.plot_param_importances(study)
fig_param_importances.show()  # 그래프 출력

# 최적화 과정 시각화
fig_optimization_history = vis.plot_optimization_history(study)
fig_optimization_history.show()  # 그래프 출력

test = pd.read_csv('c:/data/dacon/medicine/test.csv')
test.loc[:, 'Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)

test_x = np.stack(test['Fingerprint'].values)

test_y_pred = best_model.predict(test_x)

bpred = best_model.predict(test_x)

def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

# Validation 데이터로부터의 학습 모델 평가
val_y_pred = best_model.predict(val_x)
mse = mean_squared_error(pIC50_to_IC50(val_y), pIC50_to_IC50(val_y_pred))
rmse = np.sqrt(mse)

print(f'LGBM Optuna')
print(f'RMSE: {rmse}')
print(f'model.score : {best_model.score(val_x, val_y)}')


submit = pd.read_csv('c:/data/dacon/medicine/sample_submission.csv')
submit['IC50_nM'] = pIC50_to_IC50(test_y_pred)
submit.head()

submit.to_csv('c:/data/dacon/medicine/submission/lgbm_optuna.csv', index=False)
# https://dacon.io/competitions/official/236336/mysubmission

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# 학습 후 Feature Importance 확인
import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance 출력
feature_importances = best_model.feature_importances_

# 중요도 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x=np.arange(len(feature_importances)), y=feature_importances)
plt.title('Feature Importances from LightGBM')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()
