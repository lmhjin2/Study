import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
train = pd.read_csv("c:/data/dacon/icon/train.csv")
test = pd.read_csv("c:/data/dacon/icon/test.csv")

X = train.iloc[:, 2:].values
y = train["label"].values
X_test = test.iloc[:, 1:].values

# 라벨 인코딩
le = LabelEncoder()
y = le.fit_transform(y)

# 훈련/검증 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)


# Optuna 목적 함수 정의
def objective(trial):
    param = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": len(np.unique(y)),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "eta": trial.suggest_float("eta", 0.01, 0.3),  # learning_reate
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "seed": 42,
    }

    evals = [(dtrain, "train"), (dvalid, "valid")]
    model = xgb.train(
        param,
        dtrain,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    # 검증 정확도 평가
    preds = model.predict(dvalid)
    pred_labels = np.argmax(preds, axis=1)
    acc = accuracy_score(y_valid, pred_labels)
    return acc


# Optuna 스터디 생성 및 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # 시도 횟수 조절 가능

print("Best trial:")
print(study.best_trial)


# 최적 파라미터로 재학습
best_params = study.best_params
best_params.update(
    {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": len(np.unique(y)),
        "seed": 42,
    }
)

dtrain_full = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test)

final_model = xgb.train(
    best_params, dtrain_full, num_boost_round=study.best_trial.number + 50  # 조정 가능
)

# 테스트 예측
test_preds = final_model.predict(dtest)
test_labels = np.argmax(test_preds, axis=1)
test_labels_decoded = le.inverse_transform(test_labels)

# 제출 파일 저장
submission = pd.read_csv("c:/data/dacon/icon/sample_submission.csv")
submission["label"] = test_labels_decoded

import datetime

# 현재 날짜와 시간을 문자열로 변환 (예: 20250330_153045)
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

# 파일 이름 생성 (예: submission_20250330_153045.csv)
output_filename = f"c:/data/dacon/icon/output/xgb_optuna_{timestamp}.csv"

# CSV 파일 저장
submission.to_csv(output_filename, index=False, encoding="utf-8-sig")
print(f"Submission saved to {output_filename}")

# https://dacon.io/competitions/official/236459/mysubmission