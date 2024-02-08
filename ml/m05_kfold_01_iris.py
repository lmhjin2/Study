import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict

#1 데이터
x, y = load_iris(return_X_y=True)

n_splits =  10
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2 모델
# support? soft? vector machine classifier
model = SVC()
#3 훈련
scores = cross_val_score(model, x, y, cv = kfold)
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
