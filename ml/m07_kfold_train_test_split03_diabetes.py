import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_squared_log_error, accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression, ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 1)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits =  5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2
model = ElasticNetCV()

#3
scores = cross_val_score(model,x,y,cv=kfold)
#4
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

acc = accuracy_score(y_test, y_predict)
print('cross_val_precit acc:', acc)

# r2: 0.19037582699251843
# RMSE: 62.21322867804605
# RMSLE: 0.42893657582040895

# ElasticNetCV 의 정답률: 0.26655983401648775

# acc: [0.49154706 0.4308831  0.36568874 0.42604099 0.39912763 0.49807795
#  0.46417987 0.45828219 0.37906082 0.36389147]
#  평균 acc: 0.4277

# acc: [0.39761427 0.50790494 0.41064636 0.43083409 0.45718859] 
#  평균 acc: 0.4408

# acc: [0.39761427 0.50790494 0.41064636 0.43083409 0.45718859] 
#  평균 acc: 0.4408