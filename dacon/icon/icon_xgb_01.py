import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
train = pd.read_csv("c:/data/dacon/icon/train.csv")
test = pd.read_csv("c:/data/dacon/icon/test.csv")

# 이미지 시각화
random_indices = np.random.choice(len(train), 5, replace=False)
samples = train.iloc[random_indices]

fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, (idx, sample) in enumerate(samples.iterrows()):
    image_data = sample.iloc[2:].values.astype(np.uint8).reshape(32, 32)
    axes[i].imshow(image_data, cmap="gray")
    axes[i].set_title(f"Label: {sample['label']}\n ID: {sample['ID']}")
    axes[i].axis("off")
plt.tight_layout()
# plt.show()

# 피처 및 라벨 설정
X = train.iloc[:, 2:].values
y = train["label"].values
X_test = test.iloc[:, 1:].values

# 라벨 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 훈련/검증 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# XGBoost용 DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test)

# 하이퍼파라미터 설정
params = {
    "objective": "multi:softprob",
    "num_class": len(np.unique(y_encoded)),
    "max_depth": 10,
    "eta": 0.1,
    "eval_metric": "mlogloss",
    "seed": 42,
}

# 모델 학습
evals = [(dtrain, "train"), (dvalid, "valid")]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=True,
)

# 검증 정확도 평가
y_valid_pred_prob = model.predict(dvalid)
y_valid_pred = np.argmax(y_valid_pred_prob, axis=1)
valid_acc = accuracy_score(y_valid, y_valid_pred)
print(f"\nValidation Accuracy: {valid_acc:.4f}")

# 테스트 예측
y_test_pred_prob = model.predict(dtest)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)
y_test_labels = label_encoder.inverse_transform(y_test_pred)

# 제출 파일 생성
submission = pd.read_csv("c:/data/dacon/icon/sample_submission.csv")
submission["label"] = y_test_labels

import datetime

# 현재 날짜와 시간을 문자열로 변환 (예: 20250330_153045)
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

# 파일 이름 생성 (예: submission_20250330_153045.csv)
output_filename = f"c:/data/dacon/icon/output/xgb_01_{timestamp}.csv"

# CSV 파일 저장
submission.to_csv(output_filename, index=False, encoding="utf-8-sig")
print(f"Submission saved to {output_filename}")

# https://dacon.io/competitions/official/236459/mysubmission