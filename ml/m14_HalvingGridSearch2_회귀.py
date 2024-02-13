import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes
from sklearn.svm import SVC, SVR
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import time as tm

#1 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

n_splits =  5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},   # 12
    {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},    # 6
    {"C":[1,10,100,1000], "kernel":["sigmoid"],                     # 24
    "gamma":[0.01, 0.001, 0.0001], "degree":[3,4]}                  # 다더해서 42*5 = 210. kfold 5배
]

#2 모델
print("=============== HalvingGridSearchCV 시작 =======================================")
model = HalvingGridSearchCV(SVR(), parameters, 
                    # cv = kfold,
                    cv = 5 ,
                    verbose=1, 
                    refit = True, # 주석처리하면 model.best_estimator_가안먹음
                    # n_jobs=-1     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
                    # n_iter=10,  # 기본값이 10. 몇번 반복할지 정하는것 / n_iter * kfold = 시행횟수 ## 이건 RandomSearchCV에서만
                    random_state= 42 ,   # RandomizedSearchCV의랜덤값.
                    factor = 3 ,   # 3이 기본값
                    min_resources= 150 ,
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
print('r2_score:', r2_score(y_test,y_predict))
    # accuracy_score: 0.9666666666666667

y_pred_best = model.best_estimator_.predict(x_test)
            # SVC(C=1000, kernel='linear').predict(x_test)
print('최적 튠 R2:', r2_score(y_test,y_pred_best))
    # 최적 튠 ACC: 0.9666666666666667

print('걸린시간:', np.round(end_time - strat_time, 2), '초')
    # 걸린시간: 1.82 초

# import pandas as pd
# print(pd.DataFrame(model.cv_results_).transpose()) # 잘 안보이니까 dataframe에 담아서 따로 열던가 csv파일로 만들어서 보던가

# =============== HalvingGridSearchCV 시작 =======================================
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 13        // cv * 2 + @
# max_resources_: 353
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 13
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 14
# n_resources: 39
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# ----------
# iter: 2
# n_candidates: 5
# n_resources: 117
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 351
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수 :  SVR(C=1, degree=4, gamma=0.001, kernel='sigmoid')
# 최적의 파라미터 :  {'C': 1, 'degree': 4, 'gamma': 0.001, 'kernel': 'sigmoid'}
# best_score : -0.030116735641830684
# model.score : -0.044596179029908534
# r2_score: -0.044596179029908534
# 최적 튠 R2: -0.044596179029908534
# 걸린시간: 0.23 초