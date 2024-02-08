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
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = SVC(gamma=gamma, C = C)
        model.fit(x_train, y_train)
        
        score = model.score(x_test, y_test)
        
        if score > best_score:
            best_score = score
            best_parameters = {'C':C, 'gamma':gamma}

#   .format에 있는애를 {}안으로 넣음.  
#   .2f 는 소수점아래 두자리까지
#    f = float 실수
print('최고점수 :  {:.2f}'.format(best_score))
print('최적의 매개변수 : ', best_parameters)

# 최고점수 :  1.00
# 최적의 매개변수 :  {'C': 100, 'gamma': 0.001}

# return 은 def랑 세트인가봄.
