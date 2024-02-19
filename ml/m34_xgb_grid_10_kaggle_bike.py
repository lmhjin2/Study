import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 데이터 불러오기
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']
# print(x.shape)        # (10886, 11)

# 훈련 및 테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# pca = PCA(n_components=9)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)
parameters = {
    'n_estimators' : [100,200,300,400,500],
    'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1],
    'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'gamma' : [0, 1, 2],
    'lambda' : [0, 0.1, 0.01],
    'alpha' : [0, 0.1, 0.01]
}

# 모델 정의
xgb = XGBRegressor(random_state=0)
model = RandomizedSearchCV(xgb, parameters, cv = kfold, refit=True, n_jobs= 22 )

#3 compile train
import time as tm
start_time = tm.time()
model.fit(x_train,y_train)
end_time = tm.time()
#4 predict, test
results = model.score(x_test,y_test)
y_predict = model.predict(x_test)
y_pred_best = model.best_estimator_.predict(x_test)
r2 = r2_score(y_test, y_predict)
scores = cross_val_score(model, x_test, y_test, cv=kfold)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
print('best_score :', model.best_score_)
print('최적 튠 R2:', r2_score(y_test,y_pred_best))
print('r2:', scores, "\n 평균 r2:", round(np.mean(scores), 4))
print('model.score:', results)
print('r2:', results)
print('걸린시간:', np.round(end_time - start_time, 2), '초')


# XGBRegressor model.score 0.9943523513944535
# XGBRegressor : [0.65094614 0.02111737 0.16971405 0.12898925 0.01377312 0.00570355
#  0.00485386 0.00304845 0.00185426]

# 선택된 특성 수: 9
# 컬럼 줄인 XGBRegressor 의 정확도: 0.9943452254892629

# 떨어짐
# 최적의 파라미터 :  {'n_estimators': 500, 'max_depth': 2, 'learning_rate': 0.3, 'lambda': 0.01, 'gamma': 2, 'alpha': 0.01}
# best_score : 0.3149038567950451
# 최적 튠 R2: 0.32434166262548403
# r2: [0.07587486 0.30901561 0.15603168]
#  평균 r2: 0.1803
# model.score: 0.32434166262548403
# r2: 0.32434166262548403
# 걸린시간: 2.72 초