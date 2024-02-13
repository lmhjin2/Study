import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV
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
    {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},   # 12
    {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},    # 6
    {"C":[1,10,100,1000], "kernel":["sigmoid"],                     # 24
    "gamma":[0.01, 0.001, 0.0001], "degree":[3,4]}                  # 다더해서 42*5 = 210. kfold 5배
]

#2 모델

# model = SVC(C=1, kernel='linear', degree=3)

# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# model = GridSearchCV(SVC(), parameters, cv = kfold,
#                     verbose=1, 
#                     refit = True, # 주석처리하면 model.best_estimator_가안먹음
#                     n_jobs=-1     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
#                     )
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold,
                    verbose=1, 
                    refit = True, # 주석처리하면 model.best_estimator_가안먹음
                    # n_jobs=-1     # cpu 코어 몇개 쓸지 정하는거. -1이면 다씀
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




