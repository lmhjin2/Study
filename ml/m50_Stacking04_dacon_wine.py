# https://dacon.io/competitions/open/235610/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

path = "c:/_data/dacon/wine/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")


train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)

scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x.shape)    # (5497, 12)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

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

# model.score : 0.6454545454545455
# T acc:  0.6572727272727272
# F acc:  0.65

# VS acc:  0.6672727272727272
# VH acc:  0.6718181818181819

# model.score :  0.6627272727272727
# 스태킹 ACC : 0.6627272727272727