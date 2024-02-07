import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, train_size = 0.8, random_state = 0 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2
model = LinearSVC()

#3
st_time = tm.time()
model.fit(x_train, y_train)
e_time = tm.time()
r_time = np.round(e_time - st_time, 2)
#4
acc = model.score(x_test, y_test)
y_predict = model.predict(x_test)

accuracy = accuracy_score(y_predict, y_test)
print('acc:', acc)
print('accuracy:', accuracy)
print('run time ', r_time)
print('fetch covtype')

# RobustScaler  # MCP
# loss: 0.19374237954616547
# acc: 0.9310688972473145
# accuracy_score : 0.9310689052778327

# acc: 0.7124686970215915
# accuracy: 0.7124686970215915
# run time  157.25
