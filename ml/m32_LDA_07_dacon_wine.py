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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

path = "c:/_data/dacon/wine/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")


train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
x = scaler.fit_transform(x)
# print(x.shape)    # (5497, 12)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()  # n_components = 6
x = lda.fit_transform(x,y)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)
# 끄면 xgboost에러 키면 GradientBoost 에러
# ohe = OneHotEncoder()
# y_train = ohe.fit_transform(y_train.values.reshape(-1,1)).toarray()
# y_test = ohe.transform(y_test.values.reshape(-1,1)).toarray()

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

#2 모델
model = RandomForestClassifier()

#3 compile train
model.fit(x_train,y_train)

#4 predict, test
results = model.score(x_test,y_test)
# print(x.shape)
print(f"model.score : {results}")

# model.score : 0.6454545454545455

