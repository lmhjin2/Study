import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_squared_log_error
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression, ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

n_splits =  10
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size = 0.85, random_state = 0 )

#2
model = ElasticNetCV()

#3
scores = cross_val_score(model,x,y,cv=kfold)
#4
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))

# r2: 0.19037582699251843
# RMSE: 62.21322867804605
# RMSLE: 0.42893657582040895

# ElasticNetCV 의 정답률: 0.26655983401648775

# acc: [0.49154706 0.4308831  0.36568874 0.42604099 0.39912763 0.49807795
#  0.46417987 0.45828219 0.37906082 0.36389147]
#  평균 acc: 0.4277