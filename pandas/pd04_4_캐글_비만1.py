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


def fit_outlier(data):  
    data = pd.DataFrame(data)
    for label in data:
        series = data[label]        # data의 label이라는 컬럼의 데이터를 series에 담음
        q1 = series.quantile(0.25)  # q1 = 25퍼센트 지점  
        q3 = series.quantile(0.75)  # q3 = 75퍼센트 지점
        iqr = q3 - q1
        upper_bound = q3 + (iqr * 1.5)     # 이상치 범위 설정
        lower_bound = q1 - (iqr * 1.5)
        
        series[series > upper_bound] = np.nan   # series안에 이상치들 전부 np.nan(결측치) 처리
        series[series < lower_bound] = np.nan
        print(series.isna().sum())      # series 안에 결측치 갯수
        # series = series.interpolate()   # 결측치 interpolate()로 채우기
        data[label] = series    # 원래 위치에 덮어쓰기
        
    # data = data.fillna(data.ffill())
    # data = data.fillna(data.bfill())  
    return data
print(train_csv.isna().sum())   # 원래의 결측치 = 0
train_csv1 = fit_outlier(train_csv)
print(train_csv1.isna().sum())   # 함수 실행 이후의 결측치 == 기존의 이상치
    # 총합                             26570
    # Gender                               0
    # Age                               1074
    # Height                               4
    # Weight                               0
    # family_history_with_overweight    3744
    # FAVC                              1776
    # FCVC                                 0
    # NCP                               6052
    # CAEC                              3229
    # SMOKE                              245
    # CH2O                                 0
    # SCC                                687
    # FAF                                  0
    # TUE                                  0
    # CALC                              5692
    # MTRANS                            4071
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

imputer = IterativeImputer()
train_csv1 = imputer.fit_transform(train_csv1)

# train_csv1 = train_csv1.fillna(train_csv1.mean())
# print(train_csv1.insull().sum())
# print(np.isnan(train_csv1).sum)
# print(train_csv1.info())


x = train_csv.drop(['NObeyesdad'], axis = 1)
y = train_csv['NObeyesdad']

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state= 5 )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

n_splits =10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state = 5 )   # kfold 의 random_state는 점수에 영향 X

from sklearn.metrics import accuracy_score, r2_score
import catboost as cbt

#2
model = cbt.CatBoostClassifier(
    learning_rate = 0.08 ,
    iterations= 1000 ,
    depth= 6 ,
    l2_leaf_reg= 8 ,
    # loss_function= 'Logloss',
    # eval_metric= 'LogLoss',
    # subsample= 1 ,
    # task_type= 'GPU',
    random_seed= 0 , # 기본값 None
    # cat_features= # 기본값 None // 자동인식
    # bootstrap_type= 'Bayesian',
    # verbose = 1, 
    )

#3
model.fit(x_train, y_train)
#4
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)
acc = accuracy_score(y_test, y_predict)

# y_test = np.argmax(y_test, axis = 1)            # argmax주석해야함
# y_predict = np.argmax(y_predict, axis =1)       # argmax주석해야함
# y_submit = np.argmax(y_submit, axis=1)          # argmax주석해야함
y_submit = lae_NObeyesdad.inverse_transform(y_submit)   # 주석하면 0점.
scores = cross_val_score(model, x_test, y_test, cv = kfold)

submission_csv['NObeyesdad'] = y_submit
submission_csv.to_csv(path + "submission_0218_C_2.csv", index=False)

print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
print('results:', results)
print('acc:', acc)

# https://www.kaggle.com/c/playground-series-s4e2/overview

# 점수 : 0.90498
#  평균 acc: 0.8979
# results: 0.9171483622350675
# acc: 0.9171483622350675

# 점수 : 0.90498
# random_seed = 0
#  평균 acc: 0.8993
# results: 0.9190751445086706
# acc: 0.9190751445086706

# .dropna()
# 점수 : 0.90245
#  평균 acc: 0.8964
# results: 0.9178709055876686
# acc: 0.9178709055876686

# .fillna(train_csv1.mean())
# 점수 : 0.90245
#  평균 acc: 0.8964
# results: 0.9178709055876686
# acc: 0.9178709055876686

# KNNImputer
# 점수 : 0.90245
#  평균 acc: 0.8964
# results: 0.9178709055876686
# acc: 0.9178709055876686

# IterativeImputer
# 점수 : 0.90245
#  평균 acc: 0.8964
# results: 0.9178709055876686
# acc: 0.9178709055876686



