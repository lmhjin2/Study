 # https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

n_splits =10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

scaler = MinMaxScaler()

scaler.fit(x)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators

allAlgorithms = all_estimators(type_filter='classifier')

for name, algorithm in allAlgorithms:
    try:
        #2 모델
        model = algorithm()
        #3 훈련
        scores = cross_val_score(model, x_train, y_train, cv = kfold)
        print('============', name, '============')
        print('r2:', scores, "\n 평균 r2:", round(np.mean(scores), 4))
        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
        r2 = r2_score(y_test, y_predict)
        print('cross_val_precit r2:', r2)
    except Exception as e:
        # print(name, '에러', e)
        continue

# scaler = MinMaxScaler()
# run time: 50.65
# r2: 0.37518150868450384
# loss : [20384.2578125, 104.17037200927734, 20384.2578125, 1.7026294469833374]


# HistGradientBoostingRegressor 의 정답률: 0.3659113592085417

# acc: [0.30965479 0.36720921 0.35261475 0.34470105 0.34090175 0.35392407
#  0.37865916 0.38332136 0.40563247 0.36554041]
#  평균 acc: 0.3602

# acc: [0.35450298 0.34232768 0.34843274 0.36470743 0.36212217 0.39114579
#  0.37858529 0.35442292 0.38099563 0.36114753]
#  평균 acc: 0.3638

# r2: [0.35450298 0.34232768 0.34843274 0.36470743 0.36212217 0.39114579
#  0.37858529 0.35442292 0.38099563 0.36114753]
#  평균 r2: 0.3638
# cross_val_precit r2: 0.2848916421125739
