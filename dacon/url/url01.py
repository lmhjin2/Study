import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action="ignore")

# 학습/평가 데이터 로드
train_df = pd.read_csv("c:/data/dacon/url/train.csv")
test_df = pd.read_csv("c:/data/dacon/url/test.csv")

# '[.]'을 '.'으로 복구
train_df["URL"] = train_df["URL"].str.replace(r"\[\.\]", ".", regex=True)
test_df["URL"] = test_df["URL"].str.replace(r"\[\.\]", ".", regex=True)

# URL 길이
train_df["length"] = train_df["URL"].str.len()
test_df["length"] = test_df["URL"].str.len()

# 서브도메인 개수
train_df["subdomain_count"] = train_df["URL"].str.split(".").apply(lambda x: len(x) - 2)
test_df["subdomain_count"] = test_df["URL"].str.split(".").apply(lambda x: len(x) - 2)

# 특수 문자('-', '_', '/') 개수
train_df["special_char_count"] = train_df["URL"].apply(
    lambda x: sum(1 for c in x if c in "-_/")
)
test_df["special_char_count"] = test_df["URL"].apply(
    lambda x: sum(1 for c in x if c in "-_/")
)

## 악성 여부에 따른 분포 확인
# 변수 목록
variables = ["length", "subdomain_count", "special_char_count"]

"""
# 박스플롯
for var in variables:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=train_df, x='label', y=var)
    plt.title(f"Boxplot of {var} by is_malicious")
    plt.xlabel("is_malicious")
    plt.ylabel(var)
    plt.xticks([0, 1], ['Non-Malicious', 'Malicious'])
    plt.show()
"""

# 상관계수 계산
correlation_matrix = train_df[
    ["length", "subdomain_count", "special_char_count", "label"]
].corr()

"""
# 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
"""

# 학습을 위한 학습 데이터의 피처와 라벨 준비
X = train_df[["length", "subdomain_count", "special_char_count"]]
y = train_df["label"]

# 추론을 위한 평가 데이터의 피처 준비
X_test = test_df[["length", "subdomain_count", "special_char_count"]]

# XGBoost 학습 및 모델 저장 (K-Fold)
kf = KFold(n_splits=4, shuffle=True, random_state=42)
models = []  # 모델을 저장할 리스트
auc_scores = []

for idx, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print("-" * 40)
    print(f"Fold {idx + 1} 번째 XGBoost 모델을 학습합니다.")
    print("Epoch|         Train AUC             |         Validation AUC")

    # XGBoost 모델 학습
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="auc",
    )

    # 학습 및 Validation 성능 모니터링
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=True,
        # early_stopping_rounds=5
    )

    models.append(model)  # 모델 저장

    # 검증 데이터 예측 및 ROC-AUC 계산
    y_val_pred_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred_prob)
    print(f"Fold {idx + 1} CV ROC-AUC: {auc:.4f}")
    print("-" * 40)
    auc_scores.append(auc)

print(f"K-Fold 평균 ROC-AUC: {np.mean(auc_scores):.4f}")

# 평가 데이터 추론
# 각 Fold 별 모델의 예측 확률 계산
test_probabilities = np.zeros(len(X_test))

for model in models:
    test_probabilities += model.predict_proba(X_test)[:, 1]  # 악성 URL(1)일 확률 합산

# Soft-Voting 앙상블 (Fold 별 모델들의 예측 확률 평균)
test_probabilities /= len(models)
print("url01.py Done.")

# 결과 저장
test_df["probability"] = test_probabilities
test_df[["ID", "probability"]].to_csv(
    "c:/data/dacon/url/output/submission_01.csv", index=False
)

# https://dacon.io/competitions/official/236451/mysubmission
