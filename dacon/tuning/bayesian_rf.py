import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from bayes_opt import BayesianOptimization
#1
data = pd.read_csv('d:/data/tuning/train.csv')
# person_id 컬럼 제거
X = data.drop(['person_id', 'login'], axis=1)
y = data['login']

# GridSearchCV를 위한 하이퍼파라미터 설정
param_search_space = {
    'n_estimators': (100, 1000),
    'max_depth': (10, 100),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 8),
    'min_weight_fraction_leaf' : (0.0, 0.5),
    
}
def bayesian(n_estimators, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf):
    params={
        'n_estimators' : int(n_estimators),
        'max_depth' : int(max_depth),
        'min_samples_split' : int(min_samples_split),
        'min_samples_leaf' : int(min_samples_leaf),
        'min_weight_fraction_leaf' : float(min_weight_fraction_leaf)
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
    return auc_scores.max()


bay = BayesianOptimization(
    f=bayesian,
    pbounds=param_search_space,
    random_state=777
)

n_iter = 100
bay.maximize(init_points=5, n_iter=n_iter)

print(bay.max)

# {'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 2, 'n_estimators': 10} 
#  0.8065285569580801

# {'target': 0.8389592123769338, 'params': {'max_depth': 48.77397294833414, 
# 'min_samples_leaf': 7.408850876049278, 
# 'min_samples_split': 2.8858683899294393, 'n_estimators': 139.64049329754442}}

# {'target': 0.8966244725738397, 'params': {'max_depth': 37.491019813755145, 
# 'min_samples_leaf': 3.4490353775628773, 'min_samples_split': 3.8077851635396627, 
# 'min_weight_fraction_leaf': 0.0683359513260407, 'n_estimators': 134.46610224011096}}