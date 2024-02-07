import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# x = x.reshape(569, 6,5,1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state = 1 )

#2 
# model = Perceptron()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()

#3
model.fit(x_train, y_train)
# loss: 0.09776605665683746
# r2: 0.5982901288073281

#4 
results = model.score(x_test, y_test)   # x_test를 넣어서 predict한 값을 y_test 와 비교.
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)


print('acc:', results)
print('r2:', r2)

# scaler = RobustScaler()
# loss: [0.07344205677509308, 0.9651162624359131, 0.02196618542075157, 0.054660264402627945]
# acc:  0.9651162790697675
# r2: 0.9081097883593787

# acc: 0.9418604651162791
# r2: 0.7567873303167421



# model = Perceptron()
# acc: 0.7209302325581395
# r2: -0.16742081447963786

# model = LogisticRegression()
# acc: 0.9534883720930233
# r2: 0.8054298642533937

# model = KNeighborsClassifier()
# acc: 0.9418604651162791
# r2: 0.7567873303167421

# model = DecisionTreeClassifier()
# acc: 0.9534883720930233
# r2: 0.8054298642533937

# model = RandomForestClassifier()
# acc: 0.9302325581395349
# r2: 0.7081447963800905

## LogisticRegression = DecisionTreeClassifier

