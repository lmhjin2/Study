# https://dacon.io/competitions/open/236070/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
#1
path = "c:/_data/dacon/iris/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv.drop(['species'], axis = 1)
y = train_csv['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, 
                                train_size = 0.8, random_state = 0 )

# #2
# model = LinearSVC()

# #3
# start_time = tm.time()

# model.fit(x_train, y_train)

# end_time = tm.time()
# run_time = round(end_time - start_time, 2)

# #4 
# acc = model.score(x_test, y_test)
# y_submit = model.predict(test_csv)
# y_predict = model.predict(x_test)

# submission_csv['species'] = y_submit
# # submission_csv.to_csv(path + "submission_0112_3.csv", index=False)

# accuracy = accuracy_score(y_predict, y_test) 

# print('acc:', acc)
# print('accuracy_score :', accuracy)
# print('run time', run_time)

# acc: 1.0
# accuracy_score : 1.0

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

# Perceptron() acc 0.9583333333333334
# LogisticRegression() acc 1.0
# KNeighborsClassifier() acc 0.9583333333333334
# DecisionTreeClassifier() acc 0.9166666666666666
# RandomForestClassifier() acc 0.9166666666666666







