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

n_splits =10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
#2
model = XGBClassifier(# n_estimators = 1000 , 
                      # learning_rate = 0.1 , 
                      # max_depth = 4 ,
                      # min_child_weight= 1 ,
                      # gamma = 0 ,  
                      # subsample=0.8 ,
                      # colsample_bytree= 0.8 ,
                      # objective= 'binary:logistic' ,
                      # nthread= 4 ,
                      # seed= 27 ,
                    #   scale_pos_weight= 1 ,
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
submission_csv.to_csv(path + "submission_0214_2.csv", index=False)

print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
print('results:', results)
print('acc:', acc)

# https://www.kaggle.com/c/playground-series-s4e2/overview

# XGBoost 하이퍼 파라미터
# max_depth: 트리의 최대 깊이
# learning_rate: 학습률
# n_estimators: 생성할 트리의 수
# subsample: 각 트리에서 사용할 샘플의 비율
# colsample_bytree: 각 트리 생성 시 특성을 사용하는 비율
# objective: 목적 함수(손실 함수)
# eval_metric: 평가 지표
# gamma: 트리가 추가 분할되는 것을 제어하는 매개변수
# lambda: L2 정규화 항의 가중치
# alpha: L1 정규화 항의 가중치
# min_child_weight: 리프 노드에서 추가적인 가중치 적용을 위한 최소한의 데이터 수
# scale_pos_weight: 양성 클래스의 가중치 균형을 조정하는 매개변수
# eta: 이전 버전에서의 learning_rate에 대한 별칭
# tree_method: 트리를 구축하는 데 사용되는 방법
# max_delta_step: 각 트리의 가중치 추정에서 사용되는 최대 델타 단계 크기
# seed: 랜덤 시드
# verbosity: 출력 메시지의 세부 수준





