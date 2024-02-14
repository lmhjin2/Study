import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# 데이터 불러오기
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
x = train_csv #.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

# 훈련 및 테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

# 모델 정의
models = [DecisionTreeRegressor(random_state=0), RandomForestRegressor(random_state=0),
          GradientBoostingRegressor(random_state=0), XGBRegressor(random_state=0)]

# StratifiedKFold를 사용하여 모델 평가
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

np.set_printoptions(suppress=True)

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        print(type(model).__name__, "모델의 정확도:", results)
        
        # 특성 중요도 출력
        if hasattr(model, 'feature_importances_'):
            print("특성 중요도:", model.feature_importances_)
        
        # 선택된 특성 수 출력
        num_features_to_keep = 7
        if hasattr(model, 'feature_importances_'):
            sorted_indices = np.argsort(model.feature_importances_)[::-1]
            selected_features = sorted_indices[:num_features_to_keep]
            print("선택된 특성 수:", len(selected_features))
        
            # 선택된 특성으로 다시 모델 훈련 및 평가
            x_train_selected = x_train.iloc[:, selected_features]
            x_test_selected = x_test.iloc[:, selected_features]
            model_selected = model.__class__(random_state=0)
            model_selected.fit(x_train_selected, y_train)
            y_predict_selected = model_selected.predict(x_test_selected)
            r2_selected = r2_score(y_test, y_predict_selected)
            print("컬럼 줄인", type(model).__name__, "모델의 정확도:", r2_selected)
        
        print('\n')
    except Exception as e:
        print("에러:", e)
        continue

# DecisionTreeRegressor 모델의 정확도: 0.9999954260219989
# 특성 중요도: [0.00000004 0.         0.00000001 0.00000002 0.00000012 0.00000019
#  0.00000041 0.0000001  0.00000013 0.00000014 0.99999885]
# 선택된 특성 수: 7
# 컬럼 줄인 DecisionTreeRegressor 모델의 정확도: 0.999995218742029


# RandomForestRegressor 모델의 정확도: 0.9999955479040029
# 특성 중요도: [0.00000077 0.         0.00000003 0.00000012 0.00000135 0.00000105
#  0.00000135 0.00000062 0.00000085 0.00001133 0.99998254]
# 선택된 특성 수: 7
# 컬럼 줄인 RandomForestRegressor 모델의 정확도: 0.999995578596639


# GradientBoostingRegressor 모델의 정확도: 0.9999268871517323
# 특성 중요도: [0.         0.         0.         0.         0.         0.00000004
#  0.00000026 0.00000008 0.0000007  0.00001008 0.99998883]
# 선택된 특성 수: 7
# 컬럼 줄인 GradientBoostingRegressor 모델의 정확도: 0.9999268354296761


# XGBRegressor 모델의 정확도: 0.9998274322790675
# 특성 중요도: [0.00007893 0.00002007 0.00000432 0.00002359 0.00007181 0.00003213
#  0.00009541 0.00004531 0.00009842 0.00046093 0.9990691 ]
# 선택된 특성 수: 7
# 컬럼 줄인 XGBRegressor 모델의 정확도: 0.9998295745356865