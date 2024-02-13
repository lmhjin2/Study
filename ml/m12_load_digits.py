import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
import time as tm

datasets = load_digits()    # mnist의 원판
x = datasets.data
y = datasets.target
# print(x)
# print(y)
# print(x.shape, y.shape)    # (1797, 64) (1797,)
# print(pd.value_counts(y, sort=False))
    # 0    178
    # 1    182
    # 2    177
    # 3    183
    # 4    181
    # 5    182
    # 6    181
    # 7    179
    # 8    174
    # 9    180

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size = 0.2, random_state= 0 )


from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits =10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state = 5 )

model = RandomForestClassifier()

import time as tm
start_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()

results = model.score(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
scores = cross_val_score(model, x, y, cv = kfold)

print('model.socre', results)
print('acc',acc)
print('걸린시간:', np.round(end_time-start_time,2),'초')
print('CV_score', scores)
# cross_val_score는 kfold나 stratifiedkfold를 써줘야함
# model.socre 0.9666666666666667
# acc 0.9666666666666667
# 걸린시간: 0.15 초



