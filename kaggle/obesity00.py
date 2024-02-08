import numpy as np
import pandas as pd

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
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

y = np.array(y.values.reshape(-1,1))
y_ohe = OneHotEncoder(sparse=False).fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, stratify=y, test_size=0.2, random_state=0 )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# import warnings
# warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
#2
model = RandomForestClassifier()
#3
model.fit(x_train, y_train)
#4
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)

acc = accuracy_score(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

y_test = np.argmax(y_test, axis = 1)
y_predict = np.argmax(y_predict, axis =1)
y_submit = np.argmax(y_submit, axis=1)
y_submit = lae_NObeyesdad.inverse_transform(y_submit)

submission_csv['NObeyesdad'] = y_submit
submission_csv.to_csv(path + "submission_0208_0.csv", index=False)

print('results:', results)
print('acc:', acc)
print('r2:', r2)

# https://www.kaggle.com/c/playground-series-s4e2/overview

# results: 0.8564547206165704
# acc: 0.8564547206165704
# r2: 0.7301180374269683
