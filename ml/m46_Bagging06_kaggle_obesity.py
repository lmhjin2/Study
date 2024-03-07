import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings('ignore')


path = 'c:/_data/kaggle/Obesity_Risk/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder

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

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state= 5 )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression

#2. model
xgb = XGBClassifier()
# xgb = LogisticRegression()
model = BaggingClassifier(xgb,
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=777,
                          bootstrap=True, # 기본값, 데이터 중복 허용. (샘플링)
                        #   bootstrap=False,    # 중복 허용 안함
                          
                          )

#3 compile train
model.fit(x_train, y_train)

#4
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)
acc = accuracy_score(y_test, y_predict)
print("최종점수 : ", results)
print("acc: ", acc)

y_submit = lae_NObeyesdad.inverse_transform(y_submit)   # 주석하면 0점.

submission_csv['NObeyesdad'] = y_submit
submission_csv.to_csv(path + "submission_0219_3.csv", index=False)

# https://www.kaggle.com/c/playground-series-s4e2/overview

# y_test = np.argmax(y_test, axis = 1)            # argmax주석하면 에러
# y_predict = np.argmax(y_predict, axis =1)       # argmax주석하면 에러
# y_submit = np.argmax(y_submit, axis=1)          # argmax주석하면 에러
# y_submit_best = np.argmax(y_submit_best, axis = 1)

# 최적의 파라미터 :  {'seed': 47}
# model.score : 0.9200385356454721
# 최적 튠 ACC: 0.9200385356454721

# 최적의 파라미터 :  {'seed': 34}
# model.score : 0.9200385356454721
# 최적 튠 ACC: 0.9200385356454721

# 점수 : 0.91221
# 최적의 파라미터 :  {'seed': 315}
# model.score : 0.9210019267822736
# 최적 튠 ACC: 0.9210019267822736

# 0개 컬럼이 삭제되었을 때 Score:  0.9087186897880539
# 전부 - 
# {0: 0.0, 1: -0.0009633911368015502, 2: -0.00024084778420030428, 3: -0.0012042389210018545, 
# 4: -0.0014450867052022698, 5: -0.004576107899807336, 6: -0.00433526011560692, 
# 7: -0.033718689788053924, 8: -0.037090558766859294, 9: -0.042389210019267765, 
# 10: -0.04744701348747593, 11: -0.0664739884393063, 12: -0.08116570327552985, 
# 13: -0.09200385356454721, 14: -0.11416184971098264, 15: -0.16257225433526012}

# T acc:  0.9202793834296724
# F acc:  0.914980732177264
