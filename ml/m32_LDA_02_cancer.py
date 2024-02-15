import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    StratifiedKFold, cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators
from xgboost import XGBClassifier
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

scaler = StandardScaler()
x = scaler.fit_transform(x)
lda = LinearDiscriminantAnalysis() # n_components= 1
x = lda.fit_transform(x,y)
print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, shuffle=True, random_state= 0, stratify=y)

#2 model
model = RandomForestClassifier()

#3 compile train
model.fit(x_train,y_train)

#4 predict, test
results = model.score(x_test,y_test)
# print(x.shape)
print(f"model.score : {results}")

# model.score : 0.9671052631578947