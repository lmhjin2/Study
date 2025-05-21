import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler

# 1
data = pd.read_csv("d:/data/tuning/train.csv")
# person_id 컬럼 제거
X = data.drop(["person_id", "login"], axis=1)
y = data["login"]

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

param_search_space = {
    "n_estimators": hp.quniform("n_estimators", 100, 900, 100),
    "max_depth": hp.quniform("max_depth", 1, 9, 1),
    "min_samples_split": hp.quniform("min_samples_split", 1, 9, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 9, 1),
}

# 2 RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)
n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
# GridSearchCV 객체 생성
model = GridSearchCV(
    estimator=rf,
    param_grid=param_search_space,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    scoring="roc_auc",
)


def hyper(param_search_space):
    params = {
        "n_estimators": param_search_space["n_estimators"],
        "max_depth": param_search_space["max_depth"],
        "min_samples_split": param_search_space["min_samples_split"],
        "min_samples_leaf": param_search_space["min_samples_leaf"],
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model.fit(X, y)
    auc_scores = cross_val_score(model, X, y, cv=kfold, scoring="roc_auc")
    return auc_scores  # .max()


trial_val = Trials()
n_iter = 100

best = fmin(
    fn=hyper,
    space=param_search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10),
)
best_trial = sorted(trial_val.trials, key=lambda x: x["result"]["loss"], reverse=True)[
    0
]
best_acc = best_trial["result"]["loss"]

print(f"Best accuracy: {best_acc:.10f}")
print(best)
