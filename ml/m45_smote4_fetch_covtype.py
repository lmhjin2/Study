from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import fetch_covtype
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

x, y = fetch_covtype(return_X_y=True)
y = y-1

import warnings
warnings.filterwarnings('ignore')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    stratify=y
)

print(x_train.shape,y_train.shape)
print(np.unique(y_train,return_counts=True))

print("===== smote =====")
from imblearn.over_sampling import SMOTE
import time

st = time.time()
smote = SMOTE(random_state=47)
x_train, y_train = smote.fit_resample(x_train,y_train)
et = time.time()
print(et-st,"sec")

print(x_train.shape,y_train.shape)
print(np.unique(y_train,return_counts=True))

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

model = RandomForestClassifier()

# fit & pred
model.fit(x_train,y_train,)

result = model.score(x_test,y_test)
print("ACC: ",result)

pred = model.predict(x_test)
f1 = f1_score(y_test,pred,average='macro')
print("F1 : ",f1)

# evaluate
# 기본
# ACC:  0.9554572601395833
# F1 :  0.92239561616082

# SMOTE
# ACC:  0.9594846948873953
# F1 :  0.9364716290118454
    