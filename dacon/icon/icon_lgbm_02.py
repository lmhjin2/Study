import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
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
X = train.iloc[:, 2:].values  # 이미지 데이터 (32x32 = 1024 픽셀)
y = train["label"].values  # 분류할 대상 라벨

X_test = test.iloc[:, 1:].values  # 테스트 데이터

# 중요: 픽셀값 정규화 (0-1 범위로)
X = X / 255.0
X_test = X_test / 255.0

# 라벨을 숫자로 변환 (Label Encoding)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 하이퍼파라미터 튜닝을 위한 파라미터 그리드
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [10, 15, -1],  # -1은 제한 없음
    "num_leaves": [31, 63, 127],
    "min_child_samples": [20, 50],
}

# 기본 모델 설정
base_model = LGBMClassifier(random_state=42)

# GridSearchCV로 최적 파라미터 탐색
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

# 최적 파라미터 찾기
print("하이퍼파라미터 튜닝 시작...")
grid_search.fit(X_train, y_train)
print("최적 파라미터:", grid_search.best_params_)
print("최고 정확도:", grid_search.best_score_)

# 최적 모델로 학습
best_model = grid_search.best_estimator_

# Validation Set에 대한 예측 및 성능 평가
y_valid_pred = best_model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print("\nValidation Accuracy:", valid_accuracy)

# 테스트 세트 예측
y_pred = best_model.predict(X_test)

# 예측된 값을 원래 라벨로 변환
y_pred_labels = label_encoder.inverse_transform(y_pred)

submission = pd.read_csv("c:/data/dacon/icon/sample_submission.csv")

submission["label"] = y_pred_labels

import datetime

# 현재 날짜와 시간을 문자열로 변환 (예: 20250330_153045)
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

# 파일 이름 생성 (예: submission_20250330_153045.csv)
output_filename = f"c:/data/dacon/icon/output/lgbm_02_{timestamp}.csv"

# CSV 파일 저장
submission.to_csv(output_filename, index=False, encoding="utf-8-sig")
print(f"Submission saved to {output_filename}")

# https://dacon.io/competitions/official/236459/mysubmission