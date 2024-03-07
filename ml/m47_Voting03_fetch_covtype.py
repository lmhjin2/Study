import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier,VotingClassifier
from sklearn.utils import all_estimators
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(x.shape)    # (581012, 54)
scaler = StandardScaler()
x = scaler.fit_transform(x)

label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

#2. model
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = VotingClassifier(
    estimators= [('LR',lr),('RF',rf),('XGB', xgb)],
    voting='soft',
    # voting='hard',  # 기본값
)

#3 compile train
model.fit(x_train, y_train)

#4 predict, test
results = model.score(x_test, y_test)
print("최종점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc: ", acc)



# acc:  0.7484058070789911
# T acc:  0.8740652134626473
# F acc:  0.8669827801347642

# VS acc:  0.901078285414318
# VH acc:  0.8825589700782251