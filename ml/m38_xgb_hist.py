from sklearn.datasets import load_diabetes
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score, f1_score, roc_auc_score, mean_absolute_error
import time

#1.데이터
x,y = load_diabetes(return_X_y=True)
# 이진분류
# print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=3,
                                                    # stratify=y
                                                    )

sclaer = MinMaxScaler()
sclaer.fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

parameters = {
    'n_estimators': 400,  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': 0.08,  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': 8,  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight': 1,  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': 0.1,  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
    'subsample': 0.6,  # 각 트리마다의 관측 데이터 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bytree': 0.6,  # 각 트리 구성에 필요한 컬럼(특성) 샘플링 비율/ 디폴트 1 / 0~1
    'colsample_bylevel': 0.6, #  디폴트 1 / 0~1
    'colsample_bynode': 0.6, #  디폴트 1 / 0~1
    'reg_alpha' : 0.5,   # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제(제한) / alpha
    'reg_lambda' :   0.7,   # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제(제한) / lambda
    
}

#2. 모델 구성
# model = XGBClassifier()
model = XGBRegressor()

model.set_params(
    **parameters,
    early_stopping_rounds = 10 
                 )

# 3. 훈련
start = time.time()

model.fit(x_train, y_train,         # 여기다가 hist = 하면 그냥 get_params()랑 똑같이나옴
          eval_set = [(x_train, y_train),(x_test, y_test)],
          verbose = 1,  # true 디폴트 1 / false 디폴트 0 / verbose = n (과정을 n의배수로 보여줌)
        #   eval_metric='rmse', # 회귀 기본값  //  acc
          eval_metric='mae',  # rmsle, mape, mphe..등등  //  acc
        #   eval_metric='error',  # 이진분류  //  acc
        #   eval_metric='merror',  # 다중분류  //  error
        #   eval_metric='logloss', # 이진분류 기본값  //  acc
        #   eval_metric='auc',  # 모든분류  //  acc
        #   eval_metric='aucpr', # 아마 auc 친구  //  acc
          )
# https://xgboost.readthedocs.io/en/stable/parameter.html  <-  xgboost parameter official
# 4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
# print("파라미터 : ", model.get_params())
print("최종점수 : ", results)
print(f"R2 : {r2_score(y_test, y_predict)}")
print(f"MAE : {mean_absolute_error(y_test, y_predict)}")
# print(f"acc : {accuracy_score(y_test,y_predict)}")
# print(f"f1 : {f1_score(y_test,y_predict, average='macro')}")
# print('auc:', roc_auc_score(y_test, y_predict))

# fit 에다가 hist = 하면 그냥 get_params()랑 똑같이 나옴
hist = model.evals_result()
# print(hist)

# 실습
# 그려라
import matplotlib.pyplot as plt

train_metric = hist['validation_0']['mae']
val_metric = hist['validation_1']['mae']

plt.figure(figsize=(10,6))
epochs = range(1, len(train_metric) + 1 )

plt.plot(epochs, train_metric, label='Train')
plt.plot(epochs, val_metric, label='Validation')

plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Training and Validation MAE')
plt.legend()    # 어떤선이 뭐인지 알려면 써야함.
plt.grid()
plt.show()














