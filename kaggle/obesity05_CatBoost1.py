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
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score, StratifiedKFold, cross_validate,\
    GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler


# y = np.array(y.values.reshape(-1,1))
# y_ohe = OneHotEncoder(sparse=False).fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state= 5 )

# scaler = RobustScaler()  # 캣부스트는 스케일러 쓰나 안쓰나 결과가 같음. 뭘쓰든
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

n_splits =10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state = 5 )   # kfold 의 random_state는 점수에 영향 X

from sklearn.metrics import accuracy_score, r2_score
import catboost as cbt


#2
model = cbt.CatBoostClassifier(
    learning_rate = 0.03 ,
    iterations= 1000 ,
    depth= 6 ,
    l2_leaf_reg= 3 ,
    # loss_function= 'Logloss',
    # eval_metric= 'LogLoss',
    subsample= 1 ,
    task_type= 'GPU',
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
submission_csv.to_csv(path + "submission_0214_C_1.csv", index=False)

print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
print('results:', results)
print('acc:', acc)

# https://www.kaggle.com/c/playground-series-s4e2/overview

# Catboost 하이퍼 파라미터
# iterations: 트리의 수(반복 횟수)
# learning_rate: 학습률
# depth: 트리의 최대 깊이
# l2_leaf_reg: L2 정규화 계수
# border_count: 이산화를 위한 데이터 분할 포인트 수
# loss_function: 손실 함수
# eval_metric: 평가 지표
# random_seed: 랜덤 시드        ## 바뀜. // random_state 뭘 쓰던 상관 x
# verbose: 학습 과정의 출력 레벨
# task_type: 문제 유형 (GPU를 사용할 경우 'GPU'로 설정 가능)
# class_weights: 클래스 가중치
# cat_features: 범주형 특성의 인덱스
# early_stopping_rounds: 조기 종료를 위한 반복 횟수
# metric_period: 평가 지표를 계산하는 주기
# bootstrap_type: 부트스트랩 방식
# subsample: 샘플링 비율
# sampling_frequency: 부트스트랩 샘플링 주기
# grow_policy: 트리 성장 정책
# nan_mode: 결측치 처리 방식
# leaf_estimation_method: 리프 값 추정 방법
# feature_border_type: 특성의 경계 처리 방식
# permutation_count: 순열 특성 중요도 계산을 위한 횟수


# 점수 : 0.90498
#  평균 acc: 0.8979
# results: 0.9171483622350675
# acc: 0.9171483622350675



# learning_rate (학습률): 트리를 얼마나 빠르게 학습할지를 결정합니다. 기본값은 0.03입니다.
# iterations (반복 횟수): 트리의 수 또는 라운드 수를 결정합니다. 기본값은 1000입니다.
# depth (트리의 최대 깊이): 각 트리의 최대 깊이를 결정합니다. 기본값은 6입니다.
# l2_leaf_reg (L2 정규화 파라미터): 모델의 복잡도를 제어하기 위한 L2 정규화 파라미터입니다. 기본값은 3입니다.
# random_seed (랜덤 시드): 랜덤 시드를 설정하여 결과를 재현할 수 있도록 합니다. 기본값은 None으로, 랜덤 시드를 자동으로 설정합니다.
# loss_function (손실 함수): 모델이 최적화할 손실 함수를 지정합니다. 기본값은 'Logloss'입니다.
# eval_metric (평가 메트릭): 모델을 평가할 때 사용할 메트릭을 지정합니다. 기본값은 'Logloss'입니다.
# cat_features (범주형 특성): 범주형 특성의 인덱스를 지정합니다. 기본값은 None으로, 자동으로 인식됩니다.
# bootstrap_type (부트스트랩 방법): 트리를 훈련하는 데 사용할 부트스트랩 방법을 지정합니다. 기본값은 'Bayesian'입니다.
# subsample (샘플링 비율): 각 트리를 훈련할 때 사용할 샘플링 비율을 지정합니다. 기본값은 1입니다.
