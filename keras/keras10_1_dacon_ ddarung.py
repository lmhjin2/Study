# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

#1 데이터
path = "c:/_data\\dacon/ddarung//"
train_csv = pd.read_csv(path+"train.csv", index_col=0)   # (1459,10)
# print(train_csv)    # (1459,10)
# \ \\ / // 다 됨
test_csv = pd.read_csv(path+"test.csv", index_col=0) # (715, 9)
# print(test_csv)     # (715, 9)
submission_csv = pd.read_csv(path+"submission.csv") # (715, 2)
# print(submission_csv)     # (715, 2)

# print(train_csv.shape)  # (1459,10)
# print(test_csv.shape)   # (715, 9)
# print(submission_csv.shape) # (715, 2)  # test_csv와 submission_csv 의 열 합이 12인 이유 : id column 이 중복임

# print(train_csv.columns)
    # Index(['id', 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
    #       dtype='object')
# print(train_csv.info())
# print(test_csv.info())

# print(train_csv.describe())

    # 결측치를 0을 만들거나 평균을 넣는것도 되는데 이번엔 그냥 없앨거임
######### 결측치 처리 ##############
## 1. 제거.     isnull().sum() == isna().sum()  == 결측치 몇개 있나 확인
print(train_csv.isnull().sum())

# print(train_csv.isna().sum())
train_csv = train_csv.dropna()      # 행에 하나의 결측치라도 있으면 행 전체 삭제
# print(train_csv.isna().sum())
# print(train_csv.info())
# print(train_csv.shape)      # (1328, 10)

test_csv = test_csv.fillna(test_csv.mean())     # 결측치에 평균값을 넣는다.
# print(test_csv.info())  # 715 non-null


#################### x 와 y 를 분리 ################################
x = train_csv.drop(['count'], axis=1)
# print(x)        # (1459, 9)
y = train_csv['count']
# print(y)

random_state_value = 6
train_size_value = 0.83
# R2 :  0.6519230032127051
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=train_size_value, random_state=random_state_value)
# print(x_train.shape, x_test.shape)  # (929, 9)   (399, 9)
# print(y_train.shape, y_test.shape)  # (929,)    (399,)

#2. 모델
model = Sequential()
model.add(Dense(18, input_dim = 9))
model.add(Dense(56))
model.add(Dense(49))
model.add(Dense(43))
model.add(Dense(30))
model.add(Dense(12))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
start_time = round(time.time(), 2)
model.fit(x_train, y_train, epochs = 300, batch_size = 111)
end_time = round(time.time(), 2)
run_time = round(end_time - start_time, 2)
#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
# print(y_submit)
# print(y_submit.shape)   # (715, 1)
r2 = r2_score(y_test, y_predict)

# print(submission_csv.shape)
# print("========================================================================================")
########## submission.csv 만들기 (count컬럼에 값만 넣어주면 됨)
submission_csv['count'] = y_submit      # submission_csv 에 'count' 열에 y_submit 값을 넣어준다
# print(submission_csv)   # [715 rows x 2 columns]
# print(submission_csv.shape) # (715, 2)
submission_csv.to_csv(path+"submission_0105.csv", index = False)

print("loss :", loss)
print("R2 : ", r2)
print("run time:", run_time,"초")
print("random_state_value:", random_state_value)
print("train_size_value", train_size_value)

# epochs = 300, batch_size = 111
# loss : 2795.8056640625
# R2 :  0.6519230032127051
# run time: 2.98 초
# random_state_value: 6
# train_size_value 0.77

# epochs = 300, batch_size = 111
# loss : 2567.33154296875
# R2 :  0.6312730823438166
# run time: 2.88 초
# random_state_value: 17
# train_size_value 0.75


# epochs = 300, batch_size = 111
# loss : 2486.435302734375
# R2 :  0.6466293052779348
# run time: 2.99 초
# random_state_value: 6
# train_size_value 0.83

# # epochs = 300, batch_size = 111
# loss : 3076.512939453125
# R2 :  0.5623462724606854
# run time: 1.36 초
# random_state_value: 189
# train_size_value 0.811
















