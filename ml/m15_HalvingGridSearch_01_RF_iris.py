import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict,\
    StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import time as tm

#1 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

n_splits =  5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'n_estimators':[100,200,300], 'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split': [2,3,5,10]},
    {'min_samples_split': [2,3,5,10]}
    ]

#2 모델

# model = SVC(C=1, kernel='linear', degree=3)

# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# model = GridSearchCV(SVC(), parameters, cv = kfold,
#                     verbose=1, 
#                     refit = True, # 주석처리하면 model.best_estimator_가안먹음
#                     n_jobs=-1     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
#                     )
model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv = kfold,
                    verbose=1, 
                    refit = True, # 주석처리하면 model.best_estimator_가안먹음
                    n_jobs=-1,     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
                    # n_iter= 100,
                    # random_state=0
                    factor = 2,
                    # min_resources=100
                    )
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
print("최적의 매개변수 : ", model.best_estimator_)
    # 최적의 매개변수 :  SVC(C=1000, kernel='linear')
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
    # 최적의 파라미터 :  {'C': 1000, 'degree': 3, 'kernel': 'linear'}
print('best_score :', model.best_score_)
    # best_score : 0.9583333333333334
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.9666666666666667

y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test,y_predict))
    # accuracy_score: 0.9666666666666667

y_pred_best = model.best_estimator_.predict(x_test)
            # SVC(C=1000, kernel='linear').predict(x_test)
print('최적 튠 ACC:', accuracy_score(y_test,y_pred_best))
    # 최적 튠 ACC: 0.9666666666666667

print('걸린시간:', np.round(end_time - strat_time, 2), '초')
    # 걸린시간: 1.82 초

import pandas as pd
# print(pd.DataFrame(model.cv_results_).transpose()) # 잘 안보이니까 dataframe에 담아서 따로 열던가 csv파일로 만들어서 보던가

# Fitting 5 folds for each of 100 candidates, totalling 500 fits
# 최적의 매개변수 :  SVC(C=100, gamma=0.01)
# 최적의 파라미터 :  {'kernel': 'rbf', 'gamma': 0.01, 'C': 100}
# best_score : 0.9583333333333334
# model.score : 1.0
# accuracy_score: 1.0
# 최적 튠 ACC: 1.0
# 걸린시간: 1.89 초


# iter: 2
# n_candidates: 21
# n_resources: 120
# Fitting 5 folds for each of 21 candidates, totalling 105 fits
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=5, n_estimators=300)
# 최적의 파라미터 :  {'max_depth': 12, 'min_samples_leaf': 5, 'n_estimators': 300}
# best_score : 0.95
# model.score : 0.9666666666666667
# accuracy_score: 0.9666666666666667
# 최적 튠 ACC: 0.9666666666666667
# 걸린시간: 6.93 초
