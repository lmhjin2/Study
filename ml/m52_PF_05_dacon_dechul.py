# https://dacon.io/competitions/official/236214/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, RobustScaler
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
import warnings

warnings.filterwarnings('ignore')

path = "c:/_data/dacon/dechul/"
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

le_work_period = LabelEncoder() 
le_work_period.fit(train_csv['근로기간'])
train_csv['근로기간'] = le_work_period.transform(train_csv['근로기간'])
test_csv['근로기간'] = le_work_period.transform(test_csv['근로기간'])

le_purpose = LabelEncoder()
test_csv.iloc[34486,7] = '이사'     # 결혼 -> 이사 로 임의로 바꿈
le_purpose.fit(train_csv['대출목적'])
train_csv['대출목적'] = le_purpose.transform(train_csv['대출목적'])
test_csv['대출목적'] = le_purpose.transform(test_csv['대출목적'])

le_own = LabelEncoder()
le_own.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = le_own.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = le_own.transform(test_csv['주택소유상태'])

le_loan_period = LabelEncoder()
le_loan_period.fit(train_csv['대출기간'])
train_csv['대출기간'] = le_loan_period.transform(train_csv['대출기간'])
test_csv['대출기간'] = le_loan_period.transform(test_csv['대출기간'])

le_grade = LabelEncoder()
le_grade.fit(train_csv['대출등급'])
train_csv['대출등급'] = le_grade.transform(train_csv['대출등급'])

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']


pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)

scaler = RobustScaler()
x = scaler.fit_transform(x)
x_poly = scaler.fit_transform(x_poly)

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis()  # n_components = 6
# x = lda.fit_transform(x,y)
# print(x.shape)

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
model2.fit(xp_train,yp_train)
#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score : ', model.score(x_test,y_test))

yp_predict = model2.predict(xp_test)
p_score = model2.score(xp_test, yp_test)
print('model2.score : ', p_score)

# model.score : 0.5280101540414238
# acc: 0.8637858420354237

# 0개 컬럼이 삭제되었을 때 Score:  0.6575318756129925

# T acc:  0.8537471874459124
# F acc:  0.8509779034212196

# VS acc:  0.9824561403508771
# VH acc:  0.9824561403508771

# model.score :  0.8562856978018808
# 스태킹 ACC : 0.8562856978018808

# model.score :  0.3897917856586531
# model2.score :  0.30915416169063814