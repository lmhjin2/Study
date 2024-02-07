import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVR, LinearSVC
#1
datasets = load_wine()
x = datasets.data
y = datasets.target

x = x.astype(np.float32)
y = y.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, train_size = 0.8, random_state = 4 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
model = LinearSVC(C=100)


#3
# model.fit(x_train, y_train)

# #4
# acc = model.score(x_test, y_test)
# y_predict = model.predict(x_test)
# print('acc:', acc)

# acc = accuracy_score(y_predict, y_test)
# print('accuracy_score :', acc)
# print('wine')
# scaler = StandardScaler()
# loss: 0.018835054710507393
# acc: 1.0
# accuracy_score : 1.0

# StandardScaler

# LSTM
# loss: 0.14838989078998566
# acc: 0.9722222089767456
# accuracy_score : 0.9722222222222222

# Conv1D
# loss: 0.6062490344047546
# acc: 0.9444444179534912
# accuracy_score : 0.9444444444444444

from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model_P = Perceptron()
model_L = LogisticRegression()
model_K = KNeighborsClassifier()
model_D = DecisionTreeClassifier()
model_R = RandomForestClassifier()

models = [model_P, model_L,  model_K, model_D, model_R]

for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(model, "acc", results)

model_P = Perceptron()
model_L = LinearRegression()
model_K = KNeighborsRegressor()
model_D = DecisionTreeRegressor()
model_R = RandomForestRegressor()  

# Perceptron() acc 0.9722222222222222 =========================================================================================
# LogisticRegression() acc 1.0
# KNeighborsClassifier() acc 0.9722222222222222 =========================================================================================
# DecisionTreeClassifier() acc 0.9444444444444444
# RandomForestClassifier() acc 1.0