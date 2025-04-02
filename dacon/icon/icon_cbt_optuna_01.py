import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import optuna

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
X = train.iloc[:, 2:].values    # 이미지 데이터 (32x32 = 1024 픽셀)
y = train["label"].values         # 분류 대상
X_test = test.iloc[:, 1:].values  # 테스트 데이터

# 라벨 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.reavel())

# 훈련/검증 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ================================
# 1. Optuna Objective 함수 정의
# ================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_state": 42,
        "eval_metric": "MultiClass",
        "verbose": 0,
    }
    
    # CatBoostClassifier 생성 (verbose=0로 설정하여 내부 로그 최소화)
    model = CatBoostClassifier(**params)
    
    # 조기 종료를 위해 early_stopping_rounds 지정 (fit()에선 eval_metric은 전달하지 않음)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=10
    )
    
    # 검증 데이터 예측 및 정확도 산출
    y_valid_pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, y_valid_pred)
    
    return acc

# ================================
# 2. Optuna 스터디 수행
# ================================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # n_trials는 필요에 따라 조정

print("🎯 Best trial:")
print("  Accuracy:", study.best_trial.value)
print("  Params:", study.best_trial.params)

# ================================
# 3. 최적 파라미터로 최종 모델 학습
# ================================
# 전체 데이터(훈련+검증)를 활용해 최종 학습하거나, 훈련 데이터만 사용할 수 있습니다.
# 여기서는 훈련과 검증 데이터를 합쳐서 최종 모델을 학습합니다.
X_full = np.concatenate([X_train, X_valid], axis=0)
y_full = np.concatenate([y_train, y_valid], axis=0)

# 최적 파라미터에 추가 고정 파라미터를 업데이트
best_params = study.best_trial.params
best_params.update({
    "random_state": 42,
    "eval_metric": "MultiClass",
    "verbose": 100  # 진행률 출력 빈도 (원하는 대로 조정)
})

final_model = CatBoostClassifier(**best_params)
final_model.fit(X_full, y_full)

# ================================
# 4. 테스트 데이터 예측 및 제출 파일 생성
# ================================
y_pred = final_model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

submission = pd.read_csv("c:/data/dacon/icon/sample_submission.csv")
submission["label"] = y_pred_labels

# 타임스탬프(두 자리 연도, 월, 일, 시, 분, 초)로 파일명 생성
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
output_filename = f"c:/data/dacon/icon/output/cbt_optuna_01_{timestamp}.csv"

submission.to_csv(output_filename, index=False, encoding="utf-8-sig")
print(f"Submission saved to {output_filename}")

# https://dacon.io/competitions/official/236459/mysubmission