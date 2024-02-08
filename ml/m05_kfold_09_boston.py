# restore_best_weights 와
# save_best_only 에 대한 고찰

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

#1
datasets = load_boston()

x = datasets.data
y = datasets.target

# 라벨 인코딩. StratifiedKFold 할때만 필요
label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)
# 라벨 인코딩. StratifiedKFold 할때만 필요


n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2
model = HistGradientBoostingRegressor()

# #3
scores = cross_val_score(model, x, y, cv = kfold)
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))


#4


# HistGradientBoostingRegressor 의 정답률: 0.8396697405505027

# acc: [0.70891153 0.85141975 0.88898985 0.88517997 0.92975277 0.91125201
#  0.91297627 0.87654233 0.86300465 0.88762141]
#  평균 acc: 0.8716

# StratifiedKFold = y 라벨 인코딩 필요
# acc: [0.86940933 0.93149314 0.8635876  0.90785021 0.87027344 0.95898994
#  0.87159253 0.8201312  0.90246358 0.83439128]
#  평균 acc: 0.883