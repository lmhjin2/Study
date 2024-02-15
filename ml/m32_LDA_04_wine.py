import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#1
datasets = load_wine()
x = datasets.data
y = datasets.target

x = x.astype(np.float32)
y = y.astype(np.float32)
# print(x.shape, y.shape)   # (178, 13)

scaler = StandardScaler()
x = scaler.fit_transform(x)
lda = LinearDiscriminantAnalysis()  # n_components = 2
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