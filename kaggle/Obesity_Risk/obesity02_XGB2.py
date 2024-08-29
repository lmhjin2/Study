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
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

y = np.array(y.values.reshape(-1,1))
y_ohe = OneHotEncoder(sparse=False).fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, stratify=y, test_size=0.2, random_state= 2 )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

n_splits =10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state = 5 )   # kfold 의 random_state는 점수에 영향 X

from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
#2
model = XGBClassifier(n_estimators = 1000 , 
                      learning_rate = 0.1 , 
                      max_depth = 3 ,
                      min_child_weight= 7 ,
                      gamma = 1 ,  
                      subsample=0.8 ,
                      colsample_bytree= 0.8 ,
                      objective= 'binary:logistic' ,
                      nthread= 1 ,
                      seed= 315 ,
                      )

#3
model.fit(x_train, y_train)
#4
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)
acc = accuracy_score(y_test, y_predict)

y_test = np.argmax(y_test, axis = 1)            # argmax주석하면 에러
y_predict = np.argmax(y_predict, axis =1)       # argmax주석하면 에러
y_submit = np.argmax(y_submit, axis=1)          # argmax주석하면 에러
y_submit = lae_NObeyesdad.inverse_transform(y_submit)   # 주석하면 0점.
scores = cross_val_score(model, x_test, y_test, cv = kfold)

submission_csv['NObeyesdad'] = y_submit
submission_csv.to_csv(path + "submission_0228_3.csv", index=False)

print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
print('results:', results)
print('acc:', acc)

# https://www.kaggle.com/c/playground-series-s4e2/overview


# 점수 : 0.89884
# results: 0.8802986512524085
# acc: 0.8802986512524085
# tts_random = 2


# results: 0.8882466281310212
# acc: 0.8882466281310212
# tts = 5
