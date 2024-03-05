import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import all_estimators
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings('ignore')

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(x.shape)    # (581012, 54)
scaler = StandardScaler()
x = scaler.fit_transform(x)

label_endcoer = LabelEncoder()
y = label_endcoer.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)
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
#2 model
model = XGBClassifier(random_state=0)
model.set_params(early_stopping_rounds = 10, **parameters)

#3 compile train
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose= 0 )

#4 predict, test
results = model.score(x_test, y_test)
print("최종점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc: ", acc)

# model.score : 0.8850976308701153
# 오름
# 최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 9, 'learning_rate': 0.3, 'lambda': 0.01, 'gamma': 0, 'alpha': 0.1}
# best_score : 0.9590369417328016
# 최적 튠 ACC: 0.9620233556792854
# acc: [0.89492707 0.92414268 0.89462588 0.90959552 0.90843373]
#  평균 acc: 0.9063
# model.score: 0.9620233556792854
# acc: 0.9620233556792854
# 걸린시간: 465.52 초
# model.score : 0.9583333333333334
# 오름 
# 최적의 파라미터 :  {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.2, 'lambda': 0.1, 'gamma': 0, 'alpha': 0}
# best_score : 0.9652076074332172
# 최적 튠 ACC: 0.9611111111111111
# acc: [0.875      0.90277778 0.90277778 0.875      0.97222222]
#  평균 acc: 0.9056
# model.score: 0.9611111111111111
# acc: 0.9611111111111111
# 걸린시간: 4.27 초

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
          verbose=0
          )
    new_result = model2.score(new_x_test,y_test)
    print(f"{i}개 컬럼이 삭제되었을 때 Score: ",new_result)
    result_dict[i] = new_result - results    # 그대로 보면 숫자가 비슷해서 구분하기 힘들기에 얼마나 변했는지 체크
    
    
print(result_dict)

# 최종점수 :  0.7484058070789911
# acc:  0.7484058070789911

#  5: 0.0002667745238935648