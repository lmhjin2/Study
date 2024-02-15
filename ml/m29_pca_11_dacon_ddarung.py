# https://dacon.io/competitions/open/235576/mysubmission
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error, mean_absolute_error
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.dropna()  # 결측치 드랍.
test_csv = test_csv.fillna(test_csv.mean()) # 결측치에 평균치넣기
x = train_csv #.drop(['count'], axis=1)
y = train_csv['count']
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = StandardScaler()

# scaler.fit(x)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

#2
models = [DecisionTreeRegressor(random_state= 0), RandomForestRegressor(random_state= 0),
          GradientBoostingRegressor(random_state= 0), XGBRegressor(random_state= 0)]

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

# #4
y_submit = model.predict(test_csv)

submission_csv['count']=y_submit
submission_csv.to_csv(path+"submission_0131.csv",index=False)

# DecisionTreeRegressor 모델의 정확도: 0.9993249171489649
# 특성 중요도: [0.00000365 0.00000231 0.         0.00000127 0.00000303 0.00000095
#  0.00010541 0.00000156 0.00000325 0.99987858]
# 선택된 특성 수: 7
# 컬럼 줄인 DecisionTreeRegressor 모델의 정확도: 0.9998959726344715


# RandomForestRegressor 모델의 정확도: 0.9998649738483251
# 특성 중요도: [0.0000116  0.00003357 0.00000005 0.00003166 0.00003645 0.00000952
#  0.00003024 0.00005648 0.00003096 0.99975948]
# 선택된 특성 수: 7
# 컬럼 줄인 RandomForestRegressor 모델의 정확도: 0.9998489975992835


# GradientBoostingRegressor 모델의 정확도: 0.999900322197269
# 특성 중요도: [0.00000034 0.00000555 0.00000016 0.00000152 0.00000294 0.00000045
#  0.00000083 0.00000201 0.00000072 0.99998549]
# 선택된 특성 수: 7
# 컬럼 줄인 GradientBoostingRegressor 모델의 정확도: 0.9999002282242246


# XGBRegressor 모델의 정확도: 0.9996573304871267
# 특성 중요도: [0.00002322 0.00031825 0.00000735 0.00008117 0.00006925 0.00007944
#  0.00004593 0.00018307 0.00007511 0.9991172 ]
# 선택된 특성 수: 7
# 컬럼 줄인 XGBRegressor 모델의 정확도: 0.9997294183637274