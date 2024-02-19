from sklearn.datasets import load_digits
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import pickle
#1.데이터
x,y = load_digits(return_X_y=True)
# 다중분류
# print(x.shape)  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=3)

sclaer = MinMaxScaler()
sclaer.fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

#2. 모델 구성
path = 'c:/_data/_save/_pickle_test/'
model = pickle.load(open(path + 'm39_pickle1_save.dat', 'rb'))  # rb = read binary
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
# pickle.dump(model, open(path + 'm39_pickle1_save.dat', 'wb'))  # wb = write binary  //  저장
# model = pickle.load(open(path + 'm39_pickle1_save.dat', 'rb'))  # rb = read binary  //  불러오기
