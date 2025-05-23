import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("c:/data/dacon/icon/train.csv")
test = pd.read_csv("c:/data/dacon/icon/test.csv")

# train 데이터 이미지 시각화
# 랜덤 5개 샘플 선택
random_indices = np.random.choice(len(train), 5, replace=False)
samples = train.iloc[random_indices]

# 시각화
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

for i, (idx, sample) in enumerate(samples.iterrows()):
    image_data = sample.iloc[2:].values.astype(np.uint8).reshape(32, 32)  # 32x32 변환
    axes[i].imshow(image_data, cmap="gray")
    axes[i].set_title(f"Label: {sample['label']}\n ID: {sample['ID']}")
    axes[i].axis("off")

plt.tight_layout()
# plt.show()

# Feature(X)와 Target(y) 분리
X = train.iloc[:, 2:].values / 255.0  # 이미지 데이터 (32x32 = 1024 픽셀)
y = train["label"].values  # 분류할 대상 라벨

X_test = test.iloc[:, 1:].values / 255.0  # 테스트 데이터

# 라벨을 숫자로 변환 (Label Encoding)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.12,
    max_depth=30,
    # min_data_in_leaf=1,
    # min_gain_to_split=0.0,
    verbose=1,
    random_state=42,
)

# 모델 학습
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_metric="multi_logloss",
)

# Validation Set에 대한 예측 및 성능 평가
y_valid_pred = model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print("\nValidation Accuracy:", valid_accuracy)

y_pred = model.predict(X_test)

# 예측된 값을 원래 라벨로 변환
y_pred_labels = label_encoder.inverse_transform(y_pred)

submission = pd.read_csv("c:/data/dacon/icon/sample_submission.csv")

submission["label"] = y_pred_labels

import datetime

# 현재 날짜와 시간을 문자열로 변환 (예: 20250330_153045)
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

# 파일 이름 생성 (예: submission_20250330_153045.csv)
output_filename = f"c:/data/dacon/icon/output/best_{timestamp}.csv"

# CSV 파일 저장
submission.to_csv(output_filename, index=False, encoding="utf-8-sig")
print(f"Submission saved to {output_filename}")

# https://dacon.io/competitions/official/236459/mysubmission
