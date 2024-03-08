import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

path = 'c:/_data/kaggle/Obesity_Risk/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

lae_G = LabelEncoder()
train_csv['Gender'] = lae_G.fit_transform(train_csv['Gender'])
test_csv['Gender'] = lae_G.transform(test_csv['Gender'])

lae_fhwo = LabelEncoder()
train_csv['family_history_with_overweight'] = lae_fhwo.fit_transform(train_csv['family_history_with_overweight'])
test_csv['family_history_with_overweight'] = lae_fhwo.transform(test_csv['family_history_with_overweight'])

lae_FAVC = LabelEncoder()
train_csv['FAVC'] = lae_FAVC.fit_transform(train_csv['FAVC'])
test_csv['FAVC'] = lae_FAVC.transform(test_csv['FAVC'])

lae_CAEC = LabelEncoder()
train_csv['CAEC'] = lae_CAEC.fit_transform(train_csv['CAEC'])
test_csv['CAEC'] = lae_CAEC.transform(test_csv['CAEC'])

lae_SMOKE = LabelEncoder()
train_csv['SMOKE'] = lae_SMOKE.fit_transform(train_csv['SMOKE'])
test_csv['SMOKE'] = lae_SMOKE.transform(test_csv['SMOKE'])

lae_SCC = LabelEncoder()
train_csv['SCC'] = lae_SCC.fit_transform(train_csv['SCC'])
test_csv['SCC'] = lae_SCC.fit_transform(test_csv['SCC'])

lae_CALC = LabelEncoder()
test_csv['CALC'] = lae_CALC.fit_transform(test_csv['CALC'])
train_csv['CALC'] = lae_CALC.transform(train_csv['CALC'])

lae_MTRANS = LabelEncoder()
train_csv['MTRANS'] = lae_MTRANS.fit_transform(train_csv['MTRANS'])
test_csv['MTRANS'] = lae_MTRANS.transform(test_csv['MTRANS'])

lae_NObeyesdad = LabelEncoder()
train_csv['NObeyesdad'] = lae_NObeyesdad.fit_transform(train_csv['NObeyesdad'])

x = train_csv.drop(['NObeyesdad'], axis = 1)
y = train_csv['NObeyesdad']

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

# y = np.array(y.values.reshape(-1,1))
# y_ohe = OneHotEncoder(sparse=False).fit_transform(y)

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

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression


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

y_submit = model.predict(test_csv)

y_submit = lae_NObeyesdad.inverse_transform(y_submit)   # 주석하면 0점.

submission_csv['NObeyesdad'] = y_submit
submission_csv.to_csv(path + "submission_0307_3.csv", index=False)

# https://www.kaggle.com/c/playground-series-s4e2/overview

# y_test = np.argmax(y_test, axis = 1)            # argmax주석하면 에러
# y_predict = np.argmax(y_predict, axis =1)       # argmax주석하면 에러
# y_submit = np.argmax(y_submit, axis=1)          # argmax주석하면 에러
# y_submit_best = np.argmax(y_submit_best, axis = 1)

# 점수 : 0.91221
# 최적의 파라미터 :  {'seed': 315}
# model.score : 0.9210019267822736
# 최적 튠 ACC: 0.9210019267822736

# 0개 컬럼이 삭제되었을 때 Score:  0.9087186897880539

# T acc:  0.9202793834296724
# F acc:  0.914980732177264

# VS acc:  0.9161849710982659
# VH acc:  0.9144990366088632

# model.score :  0.9142581888246628
# 스태킹 ACC : 0.9142581888246628

# model.score :  0.821290944123314
# model2.score :  0.8501926782273603