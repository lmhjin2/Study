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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1 데이터
x, y = load_digits(return_X_y=True)
# print(x.shape, y.shape)     # 64 columns
# print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state= 0, 
    stratify=y,
)
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
#2 model
xgb = XGBClassifier(random_state=0)
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
acc = accuracy_score(y_test, y_predict)
scores = cross_val_score(model, x_test, y_test, cv=kfold)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
print('best_score :', model.best_score_)
print('최적 튠 ACC:', accuracy_score(y_test,y_pred_best))
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
print('model.score:', results)
print('acc:', acc)
print('걸린시간:', np.round(end_time - start_time, 2), '초')

# model.score : 0.9583333333333334
# 오름 
# 최적의 파라미터 :  {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.2, 'lambda': 0.1, 'gamma': 0, 'alpha': 0}
# best_score : 0.9652076074332172
# 최적 튠 ACC: 0.9611111111111111
# acc: [0.875      0.90277778 0.90277778 0.875      0.97222222]
#  평균 acc: 0.9056
# model.score: 0.9611111111111111
# acc: 0.9611111111111111
# 걸린시간: 4.27 초