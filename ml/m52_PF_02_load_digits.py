import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PolynomialFeatures
import time as tm
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
import warnings

warnings.filterwarnings('ignore')

#1 데이터
x, y = load_digits(return_X_y=True)
# print(x.shape, y.shape)     # 64 columns
# print(x.shape)
pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_poly = scaler.fit_transform(x_poly)

xp_train, xp_test, yp_train, yp_test = train_test_split(
    x_poly, y, random_state=777, train_size=0.8,
    stratify=y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
    stratify=y)

#2. model
model = LogisticRegression()
model2 = LogisticRegression()
#3. 훈련
model.fit(x_train,y_train)
model2.fit(xp_train, yp_train)
#4. 평가, 예측
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score : ', model.score(x_test,y_test))

yp_predict = model2.predict(xp_test)
p_score = model2.score(xp_test, yp_test)
print('model2.score : ', p_score)

# best_score : 0.9652076074332172
# T acc:  0.9666666666666667
# F acc:  0.9611111111111111
# VS acc:  0.9722222222222222
# VH acc:  0.9694444444444444

# model.score :  0.9833333333333333
# 스태킹 ACC : 0.9833333333333333

# model.score :  0.9694444444444444
# model2.score :  0.9805555555555555