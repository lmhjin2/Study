import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
train = pd.read_csv("c:/data/dacon/icon/train.csv")
test = pd.read_csv("c:/data/dacon/icon/test.csv")

# train 데이터 이미지 시각화 (랜덤 5개 샘플)
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

# Feature와 Target 분리
X = train.iloc[:, 2:].values  # 이미지 데이터 (32x32 = 1024 픽셀)
y = train["label"].values  # 분류할 대상 라벨
X_test = test.iloc[:, 1:].values  # 테스트 데이터

# 라벨 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 모델 생성 (생성자에서 eval_metric 지정)
model = CatBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=10,
    random_state=42,
    eval_metric="MultiClass",
    verbose=1, 
)

# 모델 학습 (fit()에서는 eval_metric을 전달하지 않음)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)])

# 검증 데이터에 대한 예측 및 평가
y_valid_pred = model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print("\nValidation Accuracy:", valid_accuracy)

# 테스트 데이터 예측
y_pred = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

submission = pd.read_csv("c:/data/dacon/icon/sample_submission.csv")
submission["label"] = y_pred_labels

import datetime

# 현재 날짜와 시간을 문자열로 변환 (예: 20250330_153045)
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

# 파일 이름 생성 (예: submission_20250330_153045.csv)
output_filename = f"c:/data/dacon/icon/output/cbt_01_{timestamp}.csv"

# CSV 파일 저장
submission.to_csv(output_filename, index=False, encoding="utf-8-sig")
print(f"Submission saved to {output_filename}")

# https://dacon.io/competitions/official/236459/mysubmission