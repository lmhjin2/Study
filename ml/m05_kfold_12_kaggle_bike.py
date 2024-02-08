 # https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_error
import time as tm
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor
#1
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path +"train.csv", index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

n_splits =  10
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

scaler = MinMaxScaler()

scaler.fit(x)
x = scaler.transform(x)
test_csv = scaler.transform(test_csv)

#2
model = HistGradientBoostingRegressor()

#3
scores = cross_val_score(model, x, y, cv = kfold)
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
#4 


# scaler = MinMaxScaler()
# run time: 50.65
# r2: 0.37518150868450384
# loss : [20384.2578125, 104.17037200927734, 20384.2578125, 1.7026294469833374]


# HistGradientBoostingRegressor 의 정답률: 0.3659113592085417

# acc: [0.30965479 0.36720921 0.35261475 0.34470105 0.34090175 0.35392407
#  0.37865916 0.38332136 0.40563247 0.36554041]
#  평균 acc: 0.3602