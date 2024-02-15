import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 데이터 불러오기
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
x = train_csv #.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']
# print(x.shape)        # (10886, 11)

# 훈련 및 테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
pca = PCA(n_components=9)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# 모델 정의
models = [DecisionTreeRegressor(random_state=0), RandomForestRegressor(random_state=0),
          GradientBoostingRegressor(random_state=0), XGBRegressor(random_state=0)]

# StratifiedKFold를 사용하여 모델 평가
# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

np.set_printoptions(suppress=True)


for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_, end='\n\n')

        # 남길 상위 특성 선택
        num_features_to_keep = 9
        sorted_indices = np.argsort(model.feature_importances_)[::-1]
        selected_features = sorted_indices[:num_features_to_keep]

        # 선택된 특성 수 출력
        print("선택된 특성 수:", len(selected_features))

        # 상위컬럼 데이터로 변환
        x_train_selected = x_train[:, selected_features]
        x_test_selected = x_test[:, selected_features]

        # 재학습, 평가
        model_selected = model.__class__(random_state=0)
        model_selected.fit(x_train_selected, y_train)
        y_predict_selected = model_selected.predict(x_test_selected)
        r2_selected = r2_score(y_test, y_predict_selected)

        # 프린트
        print("컬럼 줄인", type(model).__name__,"의 정확도:", r2_selected)
        print('\n')
    except Exception as e:
        print("에러:", e)
        continue

evr = pca.explained_variance_ratio_
print(np.cumsum(evr))

# DecisionTreeRegressor model.score 0.9761299150463644
# DecisionTreeRegressor : [0.74378727 0.01570073 0.16495155 0.05877717 0.00909148 0.00222607
#  0.00242057 0.00181891 0.00122624]

# 선택된 특성 수: 9
# 컬럼 줄인 DecisionTreeRegressor 의 정확도: 0.9757382111780228


# RandomForestRegressor model.score 0.9917526512742932
# RandomForestRegressor : [0.74141591 0.01594837 0.12713391 0.09886404 0.00862759 0.00226008
#  0.00233147 0.00177012 0.0016485 ]

# 선택된 특성 수: 9
# 컬럼 줄인 RandomForestRegressor 의 정확도: 0.9918323372341638


# GradientBoostingRegressor model.score 0.9893656450811784
# GradientBoostingRegressor : [0.73812393 0.01843706 0.12380571 0.10866087 0.00547871 0.00397474
#  0.00035901 0.00086135 0.00029864]

# 선택된 특성 수: 9
# 컬럼 줄인 GradientBoostingRegressor 의 정확도: 0.9893648831778659


# XGBRegressor model.score 0.9943523513944535
# XGBRegressor : [0.65094614 0.02111737 0.16971405 0.12898925 0.01377312 0.00570355
#  0.00485386 0.00304845 0.00185426]

# 선택된 특성 수: 9
# 컬럼 줄인 XGBRegressor 의 정확도: 0.9943452254892629


# [0.3093999  0.46874595 0.59132955 0.69278143 0.78615519 0.8634264
#  0.93220504 0.97307563 0.99862958]
