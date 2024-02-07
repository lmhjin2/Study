import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
                                            # regression이지만 의외로 분류
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
                                            
#1
datasets = load_iris()
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify= y, train_size = 0.8, random_state = 0 )

#2
# model = LinearSVC(C=100)
# C가 크면training 포인트를 정확히구분(굴곡지다), C가 작으면 직선에 가깝다.
# model = Perceptron()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()

#3
model.fit(x_train, y_train)

#4
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)

# print(y_predict)
acc = accuracy_score(y_predict, y_test)

print('model.score :', results)
print("accuracy_score :", acc)

## model.score 
# 분류모델에선 acc, 회귀에서는 r2
# 회귀에서는 LinearSVR 씀                                                                                       

# model = Perceptron()
# model.score : 0.9666666666666667
# accuracy_score : 0.9666666666666667

# model = LogisticRegression()   =========================================================================================
# model.score : 1.0
# accuracy_score : 1.0

# model = KNeighborsClassifier() =========================================================================================
# model.score : 1.0
# accuracy_score : 1.0

# model = DecisionTreeClassifier()
# model.score : 0.9666666666666667
# accuracy_score : 0.9666666666666667

# model = RandomForestClassifier()
# model.score : 0.9666666666666667
# accuracy_score : 0.9666666666666667


