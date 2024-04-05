import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler

#1
data = pd.read_csv('d:/data/tuning/train.csv')
# person_id 컬럼 제거
X = data.drop(['person_id', 'login'], axis=1)
y = data['login']
  
# GridSearchCV를 위한 하이퍼파라미터 설정
param_search_space = {
    'n_estimators': [110],
    # 'criterion' : ['gini'],
    'max_depth': [None],
    'min_samples_split': [2,17],   # 2이상의 정수 또는 0과 1사이의 실수(비율)
    'min_samples_leaf': [3,8],    # 1이상의 정수 또는 0과 0.5 사이의 실수(비율)
    'min_weight_fraction_leaf' : [0.0, 0.127773373, 0.1],     # 0.0 ~ 0.5 실수
    # 'max_features' : ['auto']               # https://dacon.io/competitions/official/236229/data
    # 'max_leaf_nodes' : [None],              # None 또는 양의 정수
    # 'min_impurity_decrease' : [0.0],          # 0.0 이상의 실수
    # 'bootstrap' : [True, False]             # 부스스트랩 샘플 사용.
    }  

#2 RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state = 42 )
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 8 )
# GridSearchCV 객체 생성
model = GridSearchCV(estimator=rf, param_grid=param_search_space, cv=kfold, n_jobs=-1, verbose=2, scoring='roc_auc')

#3 GridSearchCV를 사용한 학습
# model.fit(x_train, y_train)
model.fit(X, y)

#4 최적의 파라미터와 최고 점수 출력
best_params = model.best_params_
best_score = model.best_score_

print(best_params,'\n', best_score)

submit = pd.read_csv('d:/data/tuning/sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('c:/Study/dacon/tuning/output/0405.csv', index=False)

# {'max_depth': None, 'min_samples_leaf': 8, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 110} 
#  0.8049404576607657