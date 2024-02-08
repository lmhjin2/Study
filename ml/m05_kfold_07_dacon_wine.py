# https://dacon.io/competitions/open/235610/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier

path = "c:/_data/dacon/wine/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")


train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']
   
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

scaler = MaxAbsScaler()

scaler.fit_transform(x)
test_csv = scaler.transform(test_csv)

#2
model = RandomForestClassifier()

#3
scores = cross_val_score(model, x, y, cv = kfold)
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
#4 
# RandomForestClassifier() acc 0.6727272727272727 

# RandomForestClassifier 의 정답률: 0.6709090909090909

# acc: [0.68727273 0.65090909 0.69272727 0.70909091 0.65636364 0.68363636
#  0.68       0.69945355 0.71584699 0.66848816]
#  평균 acc: 0.6844
