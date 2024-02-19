import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, \
    KFold, StratifiedKFold, \
    GridSearchCV, RandomizedSearchCV, \
    HalvingGridSearchCV, HalvingRandomSearchCV,\
    cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1
x, y = load_breast_cancer(return_X_y=True)

# print(x.shape)  # (569, 30)
# print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))
x_train, x_test , y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
    stratify=y,
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=777)

# n_estimators : [100,200,300,400,500,1000] 디폴트 100/ 1~inf/ 정수
# learning_rate : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트0.3 / 0~1/ eta
# max_depth : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~inf/ 정수
# gamma : [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0 / 0~inf
# min_child_weight : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] / 디폴트 1 / 0~inf
# subsample : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# colsample_bytree : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# colsample_bylevel : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# colsample_bynode : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1/ 0~1
# reg_alpha : [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트0 / 0~inf / L1 절대값 가중치 규제
# / alpha
# reg_lambda : [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1 / 0~inf / L2 제곱 가중치 규제
# / lambda

parameters = {
    'n_estimators' : [100],
    'learning_rate' : [0.3],
    'max_depth' : [5],
    'gamma' : [0],
    'lambda' : [1],
    'alpha' : [0]
}
xgb = XGBClassifier(random_state = 0 )
model = GridSearchCV(xgb, parameters, cv=kfold, refit=True, n_jobs= 22 )

#3
import time as tm
start_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()

#4
from sklearn.metrics import accuracy_score
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
y_pred_best = model.best_estimator_.predict(x_test)
acc = accuracy_score(y_test, y_predict)
scores = cross_val_score(model, x_test, y_test, cv = kfold)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
print('best_score :', model.best_score_)
print('최적 튠 ACC:', accuracy_score(y_test,y_pred_best))
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
print('model.score:', results)
print('acc:', acc)
print('걸린시간:', np.round(end_time - start_time, 2), '초')

# 최적의 파라미터 :  {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.2}
# best_score : 0.9692307692307691
# model.score : 0.9824561403508771
# 최적 튠 ACC: 0.9824561403508771
# acc: [0.91304348 0.95652174 0.95652174 1.         0.90909091]
#  평균 acc: 0.947
# model.score: 0.9824561403508771
# acc: 0.9824561403508771
# 걸린시간: 2.55 초

