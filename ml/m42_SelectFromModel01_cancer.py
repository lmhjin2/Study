import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')
# np.set_printoptions()

#1. data
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators':1000,
              'learning_rate': 0.01,
              'max_depth':3,
              'gamma':0,
              'min_child_weight':0,
              'subsample':0.4,  # dropout개념과 비슷
              'colsample_bytree':0.8,
              'colsample_bylevel':0.7,
              'colsample_bynode':1,
              'reg_alpha': 0,
              'reg_lamda': 1,
              'random_state': 3377,
              'verbose' :0
              }

#2. model
model = XGBClassifier()
model.set_params(early_stopping_rounds = 10, **parameters)

#3. train
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric='logloss'
          )

#4. test, predict
results = model.score(x_test, y_test)
print("최종점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc: ", acc)

####################################################################################
print(model.feature_importances_)

# for문을 사용해서 피처가 약한놈부터 하나씩 제거해서
# 29, 28, 27 ... 1 까지

####################################################################################
feature_importances_list = list(model.feature_importances_)
# print(feature_importances_list)
feature_importances_list_sorted = sorted(feature_importances_list)
# print(feature_importances_list_sorted)  # 0번이제일 낮고 29번이 제일높음 (총 컬럼 30개)
drop_feature_idx_list = [feature_importances_list.index(feature) for feature in feature_importances_list_sorted]
print(drop_feature_idx_list)

result_dict = {}
for i in range(len(drop_feature_idx_list)): # 1바퀴에 1개, 마지막 바퀴에 29개 지우기, len -1은 30개 다지우면 안돼서
    drop_idx = drop_feature_idx_list[:i] # +1을 해준건 첫바퀴에 한개를 지워야 해서. 30개 시작 하고싶으면 i 만쓰고 위에 -1 지워주면됨
    new_x_train = np.delete(x_train, drop_idx, axis = 1)
    new_x_test = np.delete(x_test, drop_idx, axis=1)
    print(new_x_train.shape, new_x_test.shape)
    
    model2 = XGBClassifier()
    model2.set_params(early_stopping_rounds = 10, **parameters)
    model2.fit(new_x_train,y_train,
          eval_set=[(new_x_train,y_train), (new_x_test,y_test)],
          verbose=0,
          eval_metric='logloss',
          )
    new_result = model2.score(new_x_test,y_test)
    print(f"{i}개 컬럼이 삭제되었을 때 Score: ",new_result)
    result_dict[i] = new_result - results    # 그대로 보면 숫자가 비슷해서 구분하기 힘들기에 얼마나 변했는지 체크
    
    
print(result_dict)


# 19개 컬럼이 삭제되었을 때 Score:  0.956140350877193
# 19: 0.00877192982456143

# 20개 컬럼이 삭제되었을 때 Score:  0.956140350877193
# 20: 0.00877192982456143

