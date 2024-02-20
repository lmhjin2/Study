from sklearn.datasets import load_digits
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time

#1.데이터
x,y = load_digits(return_X_y=True)
# 다중분류
# print(x.shape)  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=3)

sclaer = MinMaxScaler()
sclaer.fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

parameters = {
    'n_estimators': 400,  # 부스팅 라운드의 수/ 디폴트 100/ 1 ~ inf/ 정수
    'learning_rate': 0.05,  # 학습률/ 디폴트 0.3/0~1/
    'max_depth': 8,  # 트리의 최대 깊이/ 디폴트 6/ 0 ~ inf/ 정수
    'min_child_weight': 1,  # 자식에 필요한 모든 관측치에 대한 가중치 합의 최소/ 디폴트 1 / 0~inf
    'gamma': 0.1,  # 리프 노드를 추가적으로 나눌지 결정하기 위한 최소 손실 감소/ 디폴트 0/ 0~ inf
}

#2. 모델 구성
model = XGBClassifier()

model.set_params(
    **parameters,
    early_stopping_rounds = 20                
                 )
# 3. 훈련
start = time.time()

model.fit(x_train, y_train, 
          eval_set = [(x_train, y_train),(x_test, y_test)],
          verbose = 1,  # true 디폴트 1 / false 디폴트 0 / verbose = n (과정을 n의배수로 보여줌)
          )
# https://xgboost.readthedocs.io/en/stable/parameter.html  <-  xgboost parameter official
# 4. 평가, 예측
results = model.score(x_test, y_test)
print("최종점수 : ", results)
# 최종점수 :  0.9388888888888889
#################################################################
# import pickle
# path = 'c:/_data/_save/_pickle_test/'
# pickle.dump(model, open(path + 'm39_pickle1_save.dat', 'wb'))   # wb = write binary

import joblib
path = 'c:/_data/_save/_joblib_test/'
joblib.dump(model, path + 'm40_joblib1_save.dat')
