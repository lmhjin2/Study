import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
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
    "gamma":[0.01, 0.001, 0.0001], "degree":[3,4]}                  # 다더해서 40
]

#2 모델

# model = SVC(C=1, kernel='linear', degree=3)

# Fitting 5 folds for each of 42 candidates, totalling 210 fits
model = GridSearchCV(SVC(), parameters, cv = kfold,
                    # verbose=1, 
                    # refit = True, 
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
print(pd.DataFrame(model.cv_results_).transpose()) # 잘 안보이니까 dataframe에 담아서 따로 열던가 csv파일로 만들어서 보던가

#     mean_fit_time  std_fit_time  mean_score_time  std_score_time  ... split4_test_score mean_test_score std_test_score rank_test_score
# 0        0.000000      0.000000         0.000000        0.000000  ...          0.916667        0.933333       0.042492              11   
# 1        0.000399      0.000488         0.000000        0.000000  ...          0.916667        0.933333       0.042492              11   
# 2        0.000598      0.000488         0.000000        0.000000  ...          0.916667        0.933333       0.042492              11   
# 3        0.000399      0.000488         0.000000        0.000000  ...          0.916667        0.941667       0.033333               4   
# 4        0.000399      0.000488         0.000000        0.000000  ...          0.916667        0.941667       0.033333               4   
# 5        0.000399      0.000488         0.000000        0.000000  ...          0.916667        0.941667       0.033333               4   
# 6        0.000399      0.000489         0.000000        0.000000  ...          0.916667        0.933333       0.020412              11   
# 7        0.000000      0.000000         0.000000        0.000000  ...          0.916667        0.933333       0.020412              11   
# 8        0.000000      0.000000         0.000000        0.000000  ...          0.916667        0.933333       0.020412              11   
# 9        0.000199      0.000399         0.000199        0.000399  ...          0.916667        0.958333       0.037268               1   
# 10       0.000399      0.000488         0.000000        0.000000  ...          0.916667        0.958333       0.037268               1   
# 11       0.000199      0.000399         0.000000        0.000000  ...          0.916667        0.958333       0.037268               1   
# 12       0.000598      0.000488         0.000000        0.000000  ...          0.958333        0.916667       0.052705              28   
# 13       0.000598      0.000488         0.000199        0.000399  ...          0.958333        0.916667       0.052705              28   
# 14       0.000399      0.000488         0.000000        0.000000  ...          1.000000        0.925000       0.048591              24   
# 15       0.000199      0.000399         0.000399        0.000488  ...          0.958333        0.916667       0.052705              28   
# 16       0.000399      0.000488         0.000000        0.000000  ...          0.958333        0.933333       0.033333              11   
# 17       0.000199      0.000399         0.000199        0.000399  ...          1.000000        0.925000       0.048591              24   
# 18       0.000398      0.000488         0.000000        0.000000  ...          0.750000        0.808333       0.056519              39   
# 19       0.000399      0.000488         0.000000        0.000000  ...          0.958333        0.925000       0.040825              24   
# 20       0.000399      0.000488         0.000000        0.000000  ...          0.958333        0.916667       0.052705              28   
# 21       0.000399      0.000488         0.000199        0.000399  ...          0.750000        0.808333       0.056519              39   
# 22       0.000200      0.000399         0.000199        0.000399  ...          0.958333        0.925000       0.040825              24   
# 23       0.000199      0.000399         0.000399        0.000488  ...          0.958333        0.916667       0.052705              28   
# 24       0.000199      0.000399         0.000398        0.000488  ...          0.833333        0.858333       0.072648              35   
# 25       0.000399      0.000488         0.000199        0.000398  ...          0.958333        0.933333       0.050000              11   
# 26       0.000399      0.000488         0.000000        0.000000  ...          0.958333        0.916667       0.052705              28   
# 27       0.000000      0.000000         0.000200        0.000400  ...          0.833333        0.858333       0.072648              35   
# 28       0.000199      0.000398         0.000206        0.000411  ...          0.958333        0.933333       0.050000              11   
# 29       0.000399      0.000489         0.000199        0.000399  ...          0.958333        0.916667       0.052705              28   
# 30       0.000399      0.000489         0.000199        0.000397  ...          0.708333        0.816667       0.072648              37   
# 31       0.000000      0.000000         0.000399        0.000489  ...          0.958333        0.941667       0.042492               4   
# 32       0.000200      0.000399         0.000199        0.000398  ...          0.958333        0.933333       0.050000              11   
# 33       0.000199      0.000398         0.000399        0.000488  ...          0.708333        0.816667       0.072648              37   
# 34       0.000000      0.000000         0.000199        0.000399  ...          0.958333        0.941667       0.042492               4   
# 35       0.000398      0.000488         0.000200        0.000399  ...          0.958333        0.933333       0.050000              11   
# 36       0.000200      0.000399         0.000000        0.000000  ...          0.666667        0.791667       0.083333              41   
# 37       0.000200      0.000399         0.000398        0.000488  ...          0.916667        0.933333       0.042492              11   
# 38       0.000000      0.000000         0.000200        0.000400  ...          0.958333        0.941667       0.042492               4   
# 39       0.000392      0.000480         0.000000        0.000000  ...          0.666667        0.791667       0.083333              41   
# 40       0.000199      0.000399         0.000199        0.000399  ...          0.916667        0.933333       0.042492              11   
# 41       0.000000      0.000000         0.000000        0.000000  ...          0.958333        0.941667       0.042492               4   

# [42 rows x 17 columns]







