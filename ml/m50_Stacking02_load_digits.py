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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings

warnings.filterwarnings('ignore')

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

#2. model
from catboost import CatBoostClassifier
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = StackingClassifier(
    estimators=[('LR',lr), ('RF',rf), ('XGB',xgb)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs=-1,
    cv=5,
)
#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score : ', model.score(x_test,y_test))
print('스태킹 ACC :', accuracy_score(y_test, y_pred))



# best_score : 0.9652076074332172
# T acc:  0.9666666666666667
# F acc:  0.9611111111111111
# VS acc:  0.9722222222222222
# VH acc:  0.9694444444444444

# model.score :  0.9833333333333333
# 스태킹 ACC : 0.9833333333333333