import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2
model = BaggingClassifier()
#3
scores = cross_val_score(model, x, y, cv = kfold)
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

acc = accuracy_score(y_test, y_predict)
print('cross_val_precit acc:', acc)

# BaggingClassifier 의 정답률: 0.9608874125452871

# acc: [0.96320264 0.9645279  0.96358066 0.96392489 0.96466498 0.96387326
#  0.96447565 0.96387326 0.96528459 0.96413143]
#  평균 acc: 0.9642

# acc: [0.96316822 0.96404599 0.96251355 0.96370114 0.96418306 0.9643896
#  0.9639421  0.96315038 0.96311595 0.96433796]
#  평균 acc: 0.9637

# acc: [0.96397714 0.96449348 0.96352903 0.96370114 0.96385604 0.96354624
#  0.96464777 0.96370114 0.9633397  0.96476825] 
#  평균 acc: 0.964
# cross_val_precit acc: 0.9146837861328881