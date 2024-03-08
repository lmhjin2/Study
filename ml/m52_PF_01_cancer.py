import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier,\
    StackingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

import warnings
warnings.filterwarnings('ignore')

#1. data
x, y = load_breast_cancer(return_X_y=True)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_poly = scaler.fit_transform(x_poly)

xp_train, xp_test, yp_train, yp_test = train_test_split(
    x_poly, y, random_state=777, train_size=0.8,
    stratify=y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
    stratify=y)

#2. model
model = LogisticRegression()
model2 = LogisticRegression()
#3. 훈련
model.fit(x_train,y_train)
model2.fit(xp_train,yp_train)
#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score : ', model.score(x_test,y_test))

yp_predict = model2.predict(xp_test)
p_score = model2.score(xp_test, yp_test)
print('model2.score : ', p_score)
# model.score :  1.0
# 스태킹 ACC : 1.0

# model.score :  0.9736842105263158
# model2.score :  0.9912280701754386

''' 고쳐야됨.
# 원래 데이터 그리기
plt.scatter(x[:,0], y, color='blue', label = '원데이터')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Example')
# 다항식 회귀 그래프 그리기
x_plot = np.linspace(-1, 1, 100).reshape(-1,1)
x_plot_poly = pf.transform(x_plot)
y_plot = model.predict(x_plot)
y_plot2 = model2.predict(x_plot_poly)

plt.plot(x_plot, y_plot, color='red', label='Polynomial Regression')
plt.plot(x_plot, y_plot2, color='green', label='기냥')
plt.legend()
plt.show()


'''