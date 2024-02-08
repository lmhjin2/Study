import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

#1 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2 모델
# support vector machine classifier
model = SVC()

#3 훈련
scores = cross_val_score(model, x_train, y_train, cv = kfold)
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

acc = accuracy_score(y_test, y_predict)
print('cross_val_precit acc:', acc)

# acc: [0.93333333 1.         1.         1.         1.         0.93333333
#  0.93333333 1.         1.         0.8       ]
#  평균 acc: 0.96

# acc: [1.         1.         1.         0.93333333 0.93333333 0.93333333
#  1.         1.         0.93333333 0.93333333]
#  평균 acc: 0.9667