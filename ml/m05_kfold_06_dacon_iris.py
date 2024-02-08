# https://dacon.io/competitions/open/236070/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold, cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
#1
path = "c:/_data/dacon/iris/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv.drop(['species'], axis = 1)
y = train_csv['species']

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# #2
model = LinearSVC()

# #3
scores = cross_val_score(model, x, y, cv = kfold)
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))

# #4 

# acc: 1.0
# accuracy_score : 1.0


# CalibratedClassifierCV 의 정답률: 1.0
# GaussianProcessClassifier 의 정답률: 1.0
# LinearDiscriminantAnalysis 의 정답률: 1.0
# LinearSVC 의 정답률: 1.0
# LogisticRegression 의 정답률: 1.0

# acc: [0.83333333 1.         1.         0.83333333 1.         0.83333333
#  1.         1.         1.         1.        ]
#  평균 acc: 0.95

# acc: [1.         0.91666667 1.         1.         1.         0.83333333
#  0.91666667 1.         0.91666667 1.        ]
#  평균 acc: 0.9583