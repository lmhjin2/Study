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
# print(train_csv.shape)    #(10886, 11)
'''
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
'''

# XGBRegressor model.score 0.9943523513944535
# XGBRegressor : [0.65094614 0.02111737 0.16971405 0.12898925 0.01377312 0.00570355
#  0.00485386 0.00304845 0.00185426]

# 선택된 특성 수: 9
# 컬럼 줄인 XGBRegressor 의 정확도: 0.9943452254892629

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 :", quartile_1)      # 4
    print("q2 : ", q2)                  # 7 ==median
    print("3사분위 :", quartile_3)      # 10
    iqr = quartile_3 - quartile_1       # (0% ~ 75%) - (0% ~ 25%) == (25% ~ 75%)
    print("iqr : ", iqr)                # 6
    lower_bound = quartile_1 - (iqr * 1.5)      # -5
    upper_bound = quartile_3 + (iqr * 1.5)      # 19
    return np.where((data_out>upper_bound) |    # shift + \ == |   // or 과 같은뜻. python 문법
                    (data_out<lower_bound))     # np.where 이면 위치 반환해줌
    # 19보다 크거나 -5보다 작은놈들의 위치를 찾는다
outliers_loc = outliers(train_csv)
print("이상치의위치 :", outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(train_csv)
plt.show()
print(len(outliers_loc[0]))
# 이상치 위치 전부 뽑는 코드
# for i in range(len(outliers_loc[0])):
#     print("이상치 위치", outliers_loc[0][i], ",", outliers_loc[1][i])
# 1사분위 : 1.0
# q2 :  12.3
# 3사분위 : 37.88
# iqr :  36.88
# 이상치의위치 : (array([   13,    14,    15, ..., 10883, 10884, 10884], dtype=int64), array([10, 10, 10, ..., 10,  9, 10], dtype=int64))

