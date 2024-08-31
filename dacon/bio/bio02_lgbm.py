import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

# 데이터 로드
train = pd.read_csv("c:/data/dacon/bio/train.csv")
test = pd.read_csv("c:/data/dacon/bio/test.csv")

# 레이블 인코딩
le_subclass = LabelEncoder()
train['SUBCLASS'] = le_subclass.fit_transform(train['SUBCLASS'])

# for i, label in enumerate(le_subclass.classes_):
#     print(f"기존 레이블 : {label}, 변환 후 : {i}")

# 특성과 타겟 분리
X = train.drop(columns=['SUBCLASS', 'ID'])
y_subclass = train['SUBCLASS']

# 범주형 변수 인코딩
categorical_columns = X.select_dtypes(include=['object', 'category']).columns
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = X.copy()
X_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

# 학습 데이터와 검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y_subclass, test_size=0.2, random_state=42)

# XGBoost 모델 정의
model = LGBMClassifier(
    n_estimators=100,
    num_leaves = 64,
    learning_rate=0.1,
    # max_depth=6,
    random_state=42,
    early_stopping_round = 50
)

## LGBM 하이퍼 파라미터
# num_leaves: 트리에 존재할 수 있는 최대 잎의 수
# learning_rate: 학습률
# n_estimators: 생성할 트리의 수
# subsample: 각 트리에서 사용할 샘플의 비율
# colsample_bytree: 각 트리 생성 시 특성을 사용하는 비율
# objective: 목적 함수(손실 함수)
# metric: 평가 지표
# max_depth: 트리의 최대 깊이
# min_child_samples: 리프 노드에 필요한 최소 데이터 수
# lambda_l1: L1 정규화 항의 가중치
# lambda_l2: L2 정규화 항의 가중치
# min_split_gain: 분할을 수행하기 위한 최소 이득
# boosting_type: 부스팅 유형
# early_stopping_rounds: 조기 종료를 위한 반복 횟수
# verbose: 출력 메시지의 세부 수준
# seed: 랜덤 시드

# 모델 학습 (검증 데이터 포함)
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(X_train, y_train, eval_set=eval_set)

# 테스트 데이터 처리 및 예측
results = model.score(X_val, y_val)
print("model.score", results)

test_X = test.drop(columns=['ID'])
test_X_encoded = test_X.copy()
test_X_encoded[categorical_columns] = ordinal_encoder.transform(test_X[categorical_columns])
predictions = model.predict(test_X_encoded)

# 원래 레이블로 변환
original_labels = le_subclass.inverse_transform(predictions)

# 제출 파일 생성
submission = pd.read_csv("c:/data/dacon/bio/sample_submission.csv")
submission["SUBCLASS"] = original_labels
submission.to_csv('c:/data/dacon/bio/submission/bio_02_lgbm.csv', encoding='UTF-8-sig', index=False)
# https://dacon.io/competitions/official/236355/mysubmission

# [15]    training's multi_logloss: 0.971804      valid_1's multi_logloss: 2.24667
# model.score 0.32473811442385175
