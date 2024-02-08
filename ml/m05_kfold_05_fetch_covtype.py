import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = RobustScaler()

scaler.fit_transform(x)


#2
model = BaggingClassifier()
#3
scores = cross_val_score(model, x, y, cv = kfold)
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))

# BaggingClassifier 의 정답률: 0.9608874125452871

# acc: [0.96320264 0.9645279  0.96358066 0.96392489 0.96466498 0.96387326
#  0.96447565 0.96387326 0.96528459 0.96413143]
#  평균 acc: 0.9642

