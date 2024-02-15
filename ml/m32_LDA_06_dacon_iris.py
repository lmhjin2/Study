# https://dacon.io/competitions/open/236070/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
#1
path = "c:/_data/dacon/iris/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv #.drop(['species'], axis = 1)
y = train_csv['species']
# print(x.shape)        # (120,5)
scaler = StandardScaler()
x = scaler.fit_transform(x)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()  # n_components = 1
x = lda.fit_transform(x,y)
print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

#2 model
model = RandomForestClassifier()

#3 compile train
model.fit(x_train,y_train)

#4 predict, test
results = model.score(x_test,y_test)
# print(x.shape)
print(f"model.score : {results}")

# model.score : 1.0