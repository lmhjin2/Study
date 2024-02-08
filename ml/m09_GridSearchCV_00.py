import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

#1 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

# 2중 for문

best_score = 0
for gamma in [0.001, 0.01, 0.1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 10, 100]:
        model = SVC(gamma=gamma, C = C)
        model.fit(x_train, y_train)
        
        score = model.score(x_test, y_test)
        if score > best_score:
            best_score = score
            best_parameters = {'C':C, 'gamma':gamma}
        