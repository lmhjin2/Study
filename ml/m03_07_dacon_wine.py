# https://dacon.io/competitions/open/235610/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR

path = "c:/_data/dacon/wine/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")


train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, 
                                    train_size = 0.8, random_state = 0 )
   
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2
# model = LinearSVC()

# #3
# # import datetime

# # date = datetime.datetime.now()
# # date = date.strftime("%m%d_%H%M")   # 월일_시분

# # path1 = "c:/_data/_save/MCP/k28/10/"
# # filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
# # filepath = "".join([path1, 'k28_', date, '_', filename])

# start_time = tm.time()

# model.fit(x_train, y_train)

# end_time = tm.time()
# run_time = round(end_time - start_time, 2)

# #4 
# acc = model.score(x_test, y_test)
# y_submit = model.predict(test_csv)
# y_predict = model.predict(x_test)

# submission_csv['quality'] = y_submit
# # submission_csv.to_csv(path + "submission_0116_3.csv", index=False)

# accuracy = accuracy_score(y_predict, y_test) 

# print('acc:', acc)
# print('accuracy_score :', accuracy)
# print("run time:", run_time)
# print('dacon wine')

# scaler = MaxAbsScaler()
# loss: 1.1051563024520874
# acc: 0.5363636612892151
# accuracy_score : 0.5363636363636364

# LSTM
# loss: 1.1040036678314216
# acc: 0.5363636612892151
# accuracy_score : 0.5363636363636364
# run time: 23.48

# Conv1D
# loss: 1.0907200574874878
# acc: 0.5381818413734436
# accuracy_score : 0.5381818181818182
# run time: 27.18

# acc: 0.5172727272727272
# accuracy_score : 0.5172727272727272
# run time: 0.07
# dacon wine

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

# Perceptron() acc 0.4818181818181818
# LogisticRegression() acc 0.5236363636363637
# KNeighborsClassifier() acc 0.5463636363636364
# DecisionTreeClassifier() acc 0.5918181818181818
# RandomForestClassifier() acc 0.6727272727272727 =========================================================================================