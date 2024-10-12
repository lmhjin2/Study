import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
import optuna

# 데이터 로드
train = pd.read_csv("c:/data/dacon/bio/train.csv")
test = pd.read_csv("c:/data/dacon/bio/test.csv")

# 레이블 인코딩
le_subclass = LabelEncoder()
train['SUBCLASS'] = le_subclass.fit_transform(train['SUBCLASS'])

# for i, label in enumerate(le_subclass.classes_):
#     print(f"기존 레이블 : {label}, 변환 후 : {i}")

# 특성과 타겟 분리
X = train.drop(columns=['SUBCLASS', 'ID'])
y_subclass = train['SUBCLASS']

# 범주형 변수 인코딩
categorical_columns = X.select_dtypes(include=['object', 'category']).columns
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = X.copy()
X_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

# 학습 데이터와 검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y_subclass, test_size=0.2, random_state=42, 
                                                #   shuffle=True
                                                  )

# XGBoost 모델 정의
def objective(trial):
    # StratifiedKFold 설정
    n_splits = 2
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 하이퍼파라미터 제안
    param = {
        # 'objective': 'multi:softmax',
        # 'num_class': len(le_subclass.classes_),  # multi-class classification을 위해 추가
        'eval_metric': 'mlogloss',  # XGBoost에서의 metric 지정
        'n_estimators': 100,
        'learning_rate': trial.suggest_float('learning_rate', 0.14, 0.16), # 기본 0.3
        # 'max_depth': trial.suggest_int('max_depth', 1, 100), # 기본 6
        # 'max_bin': trial.suggest_int('max_bin', 128, 512), # 기본 256
        # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 50), # 기본 1
        # 'subsample': trial.suggest_float('subsample', 0.7, 1.0), # 기본 1.0
        # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0), # 기본 1.0
        # 'gamma': trial.suggest_float('gamma', 0, 1), # 기본 0
        # 'reg_alpha': trial.suggest_float('reg_alpha', 0, 50), # 기본 0
        # 'reg_lambda': trial.suggest_float('reg_lambda', 0, 10), # 기본 1
        # 'early_stopping_rounds': 10
    }
    
    rmses = []
    for train_idx, valid_idx in kfold.split(X_train, y_train):
        x_train_fold, x_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]  # 수정 부분
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]  # 수정 부분
        
        model = xgb.XGBClassifier(**param, seed=42)
        model.fit(x_train_fold, y_train_fold, eval_set=[(x_valid_fold, y_valid_fold)], verbose=True)
        
        preds = model.predict(x_valid_fold)
        rmse = np.sqrt(mean_squared_error(y_valid_fold, preds))
        rmses.append(rmse)
    
    # 최소 RMSE 반환
    min_rmse = np.min(rmses)
    return min_rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10) # 10회의 시도로 최적화

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print(f'Best RMSE Score: {study.best_trial.value}')

trial = study.best_trial
best_params = study.best_trial.params

best_model = xgb.XGBClassifier(**best_params, seed=42)
best_model.fit(X_train, y_train)

# optuna.visualization.plot_param_importances(study)      # 파라미터 중요도 확인 그래프
# optuna.visualization.plot_optimization_history(study)   # 최적화 과정 시각화

import optuna.visualization as vis
# 파라미터 중요도 확인 그래프
fig_param_importances = vis.plot_param_importances(study)
fig_param_importances.show()  # 그래프 출력

# # 최적화 과정 시각화
# fig_optimization_history = vis.plot_optimization_history(study)
# fig_optimization_history.show()  # 그래프 출력

# 테스트 데이터 처리 및 예측
results = best_model.score(X_val, y_val)

print(f'xgb_optuna')
print("model.score : ", results)

test_X = test.drop(columns=['ID'])
test_X_encoded = test_X.copy()
test_X_encoded[categorical_columns] = ordinal_encoder.transform(test_X[categorical_columns])
# bpred = best_model.predict(test_X_encoded)
predictions = best_model.predict(test_X_encoded)

# 원래 레이블로 변환
original_labels = le_subclass.inverse_transform(predictions)

# 제출 파일 생성
submission = pd.read_csv("c:/data/dacon/bio/sample_submission.csv")
submission["SUBCLASS"] = original_labels
submission.to_csv('c:/data/dacon/bio/submission/xgb_optuna.csv', encoding='UTF-8-sig', index=False)
# https://dacon.io/competitions/official/236355/mysubmission

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

