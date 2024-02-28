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
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

# y = np.array(y.values.reshape(-1,1))
# y_ohe = OneHotEncoder(sparse=False).fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state= 5 )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

n_splits =10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state = 2 )

from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

parameters = [{'seed':range(1013,1014,1)}]

model = GridSearchCV(XGBClassifier(n_estimators = 1000 , 
                      learning_rate = 0.1 , 
                      max_depth = 3 ,
                      min_child_weight= 7 ,
                      gamma = 1 ,  
                      subsample=0.8 ,
                      colsample_bytree= 0.8 ,
                      objective= 'binary:logistic' ,
                      nthread= 1 ,
                    #   seed= 3 ,   # 찾는중
                    #   tree_method = 'hist' ,      # xgb-gpu
                    #   device = 'cuda' ,           # xgb-gpu 
                    # conda install -c conda-forge xgboost-gpu <- cmd에서 gpu버전 xgboost설치해야함
                      # scale_pos_weight= 1 , # 양수데이터가 적을때 양수 데이터 중요도 올리기. 10 = 10배
                      ), parameters, cv=kfold, refit=True, n_jobs=-1 )

#3
import time as tm
start_time = tm.time()
model.fit(x_train, y_train)
end_time = tm.time()
#4
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
y_pred_best = model.best_estimator_.predict(x_test)
y_submit = model.predict(test_csv)
y_submit_best = model.best_estimator_.predict(test_csv)
acc = accuracy_score(y_test, y_predict)

# y_test = np.argmax(y_test, axis = 1)            # argmax주석하면 에러
# y_predict = np.argmax(y_predict, axis =1)       # argmax주석하면 에러
# y_submit = np.argmax(y_submit, axis=1)          # argmax주석하면 에러
# y_submit_best = np.argmax(y_submit_best, axis = 1)

y_submit = lae_NObeyesdad.inverse_transform(y_submit)   # 주석하면 0점.
y_submit_best = lae_NObeyesdad.inverse_transform(y_submit_best)

scores = cross_val_score(model, x_test, y_test, cv = kfold)

submission_csv['NObeyesdad'] = y_submit
submission_csv.to_csv(path + "submission_0228_3.csv", index=False)

submission_csv['NObeyesdad'] = y_submit_best
submission_csv.to_csv(path + "submission_b_0228_1.csv", index=False)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)     # 내가 선택한 놈만 나옴
print('best_score :', model.best_score_)
print('model.score :', model.score(x_test, y_test))
print('최적 튠 ACC:', accuracy_score(y_test,y_pred_best))
print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
print('model.score:', results)
print('acc:', acc)
print('걸린시간:', np.round(end_time - start_time, 2), '초')

# https://www.kaggle.com/c/playground-series-s4e2/overview

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
