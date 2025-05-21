import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# 데이터 로드
train = pd.read_csv("c:/data/dacon/bio/train.csv")
test = pd.read_csv("c:/data/dacon/bio/test.csv")

# 레이블 인코딩
le_subclass = LabelEncoder()
train["SUBCLASS"] = le_subclass.fit_transform(train["SUBCLASS"])

# for i, label in enumerate(le_subclass.classes_):
#     print(f"기존 레이블 : {label}, 변환 후 : {i}")

# 특성과 타겟 분리
X = train.drop(columns=["SUBCLASS", "ID"])
y_subclass = train["SUBCLASS"]

# 범주형 변수 인코딩
categorical_columns = X.select_dtypes(include=["object", "category"]).columns
ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_encoded = X.copy()
X_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

# 학습 데이터와 검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y_subclass, test_size=0.2, random_state=42
)

# XGBoost 모델 정의
model = xgb.XGBClassifier(
    n_estimators=1001,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss",
)

# 모델 학습 (검증 데이터 포함)
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# 테스트 데이터 처리 및 예측
results = model.score(X_val, y_val)
print("model.score", results)

test_X = test.drop(columns=["ID"])
test_X_encoded = test_X.copy()
test_X_encoded[categorical_columns] = ordinal_encoder.transform(
    test_X[categorical_columns]
)
predictions = model.predict(test_X_encoded)

# 원래 레이블로 변환
original_labels = le_subclass.inverse_transform(predictions)

# 제출 파일 생성
submission = pd.read_csv("c:/data/dacon/bio/sample_submission.csv")
submission["SUBCLASS"] = original_labels
submission.to_csv(
    "c:/data/dacon/bio/submission/bio_03.csv", encoding="UTF-8-sig", index=False
)
# https://dacon.io/competitions/official/236355/mysubmission
