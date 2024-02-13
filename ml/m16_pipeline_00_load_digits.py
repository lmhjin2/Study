import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import time as tm
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

#1 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

n_splits =  5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},   # 12
    {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},    # 6
    {"C":[1,10,100,1000], "kernel":["sigmoid"],                     # 24
    "gamma":[0.01, 0.001, 0.0001], "degree":[3,4]}                  # 다더해서 42*5 = 210. kfold 5배
]

#2 모델
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
strat_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
# print("최적의 매개변수 : ", model.best_estimator_)
#     # 최적의 매개변수 :  SVC(C=1000, kernel='linear')
# print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
#     # 최적의 파라미터 :  {'C': 1000, 'degree': 3, 'kernel': 'linear'}
# print('best_score :', model.best_score_)
    # best_score : 0.9583333333333334
print('model.score :', model.score(x_test, y_test))
    # model.score : 0.9666666666666667

y_predict = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_test,y_predict))
    # accuracy_score: 0.9666666666666667

# y_pred_best = model.best_estimator_.predict(x_test)
            # SVC(C=1000, kernel='linear').predict(x_test)
# print('최적 튠 ACC:', accuracy_score(y_test,y_pred_best))
    # 최적 튠 ACC: 0.9666666666666667

print('걸린시간:', np.round(end_time - strat_time, 2), '초')
    # 걸린시간: 1.82 초

# import pandas as pd
# print(pd.DataFrame(model.cv_results_).transpose()) # 잘 안보이니까 dataframe에 담아서 따로 열던가 csv파일로 만들어서 보던가


# 최적의 매개변수 :  SVC(C=100, gamma=0.001)
# 최적의 파라미터 :  {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
# best_score : 0.9444444444444444
# model.score : 1.0
# accuracy_score: 1.0
# 최적 튠 ACC: 1.0
# 걸린시간: 0.16 초

# import sklearn as sk
# print(sk.__version__) # 1.1.3 근데 아직도 experimental임. 한국어로 실험실 옵션이라 생각하면 될듯?

# =============== HalvingGridSearchCV 시작 =======================================
# n_iterations: 3           // 반복횟수: 3
# n_required_iterations: 4  // iteration 4번 필요
# n_possible_iterations: 3  // 가능한 iteration : 3
# min_resources_: 100       // cv * 2 * '라벨의 갯수' + @
# max_resources_: 1437      // train 데이터 갯수
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 42      // 42개의 후보. == parameter의 시행횟수
# n_resources: 100          // cv * 2 * '라벨의 갯수'
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 14      // iter 0 기준 상위 'factor' 분의 1
# n_resources: 300      // iter 0 * 'factor'
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# ----------
# iter: 2
# n_candidates: 5       // iter 1 기준 상위 'factor' 분의 1, 아마도 반올림? 버림은 아님
# n_resources: 900      // iter 1 * 'factor' == iter0 * 'factor'**2
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# 최적의 매개변수 :  SVC(C=10, gamma=0.0001)
# 최적의 파라미터 :  {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
# best_score : 0.9844009931719429
# model.score : 0.9833333333333333
# accuracy_score: 0.9833333333333333
# 최적 튠 ACC: 0.9833333333333333
# 걸린시간: 0.9 초

# pipeline
# model.score : 0.9638888888888889
# accuracy_score: 0.9638888888888889
# 걸린시간: 0.16 초