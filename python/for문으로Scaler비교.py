import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import warnings

warnings.filterwarnings('ignore')

#1
datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify= y, train_size = 0.8, random_state = 0 )


Scalers = [MinMaxScaler(), StandardScaler(), MaxAbsScaler(), RobustScaler()]

for scaler in Scalers:
    scaler.fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_test)
    #2
    model = LinearSVC(C=100)
    # C가 크면training 포인트를 정확히구분(굴곡지다), C가 작으면 직선에 가깝다.
    #3
    model.fit(x_train, y_train)
    #4
    results = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    # acc = accuracy_score(y_predict, y_test)

    # print(type(scaler).__name__)
    # print('model.score :', results)
    # print("accuracy_score :", acc)
    print(f"[{type(scaler).__name__}] model acc:", results)
