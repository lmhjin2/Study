import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, shuffle=True, random_state= 0, stratify=y)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2 
model = BaggingClassifier()

#3
scores = cross_val_score(model, x_train, y_train, cv=kfold)

#4 


from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

acc = accuracy_score(y_test, y_predict)
print('cross_val_precit acc', acc)

# BaggingClassifier 의 정답률: 0.9651162790697675

# acc: [0.96491228 0.94736842 0.98245614 0.96491228 0.94736842 0.92982456
#  0.96491228 0.94736842 0.96491228 0.89285714]
#  평균 acc: 0.9507

# acc: [0.92982456 0.92982456 0.9122807  0.98245614 0.94736842 0.96491228
#  0.92982456 0.96491228 0.94736842 0.98214286]
#  평균 acc: 0.9491

# acc: [0.91666667 0.91666667 0.91666667 0.81818182 1.         0.90909091
#  0.90909091 1.         1.         0.90909091]
#  평균 acc: 0.9295
# cross_val_precit acc 0.9276315789473685