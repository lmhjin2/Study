import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna

#1
data = pd.read_csv('d:/data/tuning/train.csv')
# person_id 컬럼 제거
X = data.drop(['person_id', 'login'], axis=1)
y = data['login']

def objective(trial):
    # 데이터 로드 및 준비
    data = pd.read_csv('d:/data/tuning/train.csv')
    X = data.drop(['person_id', 'login'], axis=1)
    y = data['login']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 탐색할 하이퍼파라미터 지정
    n_estimators = trial.suggest_int('n_estimators', 100, 200)
    max_depth = trial.suggest_int('max_depth', 1, 100)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)

    # 모델 생성 및 훈련
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    preds = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, preds)
    
    return auc_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200) # n_trials는 시도할 횟수

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print(f'Best ROC-AUC Score: {study.best_trial.value}')

# 최적의 하이퍼파라미터 적용
best_params = study.best_trial.params
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X, y)




submit = pd.read_csv('d:/data/tuning/sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('c:/Study/dacon/tuning/output/0321_opt.csv', index=False)






# {'max_depth': None, 'min_samples_leaf': 8, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 110} 
#  0.8049404576607657
# n_estimators,criterion,max_depth,min_samples_split,min_samples_leaf,
# min_weight_fraction_leaf,max_features,max_leaf_nodes,min_impurity_decrease,bootstrap
