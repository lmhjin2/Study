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
    'n_estimators': [80,140,200,1000],
    'max_depth': [1,15,49, 50],
    'min_samples_split': [1,3,6],
    'min_samples_leaf': [1,3,5,7,9]}
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42,
#                                                     # shuffle=True
#                                                     )

#2 RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)
n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 42 )
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

submit.to_csv('c:/Study/dacon/tuning/output/0313_2.csv', index=False)


# {'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 50} 0.7474650215816369

# {'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 2, 'n_estimators': 10} 
#  0.8065285569580801