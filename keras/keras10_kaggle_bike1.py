# https://www.kaggle.com/competitions/bike-sharing-demand/data
# 두개 이상은 list
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import time as tm

#1. 데이터
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path + "train.csv", index_col = 0)
# print(train_csv)    # (10886, 11)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
# print(test_csv.shape)     # (6493, 8)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
# print(submission_csv)       # (6493, 2)

# 결측치 확인 / 했는데 없대요
# print(train_csv.isna().sum())   # print(train_csv.isnull().sum())

x = train_csv.drop(['casual', 'registered', 'count'], axis = 1)
# print(x)    # (10886, 8)
y = train_csv['count']
# print(y)    # (10886,)

random_state_value = 1
train_size_value = 0.74
# MSE: 1.3773208856582642
# RMSE: 165.2521729059214
# RMSLE: 1.173593080617437


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    train_size = train_size_value, random_state = random_state_value)

# x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.53, random_state = 2)
# validation(검증용 데이터) 를 넣고 싶을 때 윗줄 코드 추가

# print(x_train.shape, x_test.shape)  # (7620, 10), (3266, 10)
# print(y_train.shape, y_test.shape)  # (7620,), (3266,)

#2. 모델
model = Sequential()
model.add(Dense(16, input_dim = 8, activation = 'relu'))    # relu 는 0이하는 0으로, 양수는 그대로 뽑아내는것
model.add(Dense(32, activation = 'relu'))
model.add(Dense(24, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))

#3. 컴파일 훈련
model.compile(loss='msle', optimizer='adam') 
start_time = tm.time()                          # validation_data = (x_val, y_val) / validation(검증용 데이터) 를 넣고 싶을 때 아래 fit의 ()안에 이거 쓰면 됨
model.fit(x_train, y_train, epochs = 1000, batch_size = 381, verbose = 2) # train_size 0.7 기준 x_train (7620, 8) / 0.74 기준 x_train (8055, 8)
# verbose = 0 : 침묵
# verbose = 1 : 원래 보던 그 epoch. default값
# verbose = 2 : 프로그레스바 삭제
# verbose = 3 : epoch 만 나옴
# verbose 0,1,2 제외 나머지 : epoch 만 나옴
end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)    # 원래 하던거. 근데 count에 음수가 나옴;;
# y_submit = abs(model.predict(test_csv)) # 그냥 전부 절대값을 씌웠음. 근데 이거 소용없음
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
submission_csv['count'] = y_submit  # submission_csv 에 'count' 열에 y_submit 값을 넣어줌
# print(submission_csv)   # (6493, 2)
submission_csv.to_csv(path + "submission_0109.csv", index = False)

print("음수갯수 :", submission_csv[submission_csv['count']<0].count())  ## 진짜 중요함 ##
print("R2: ", r2)
print("run time: ", run_time, "초")
print("random_state_value: ", random_state_value)
print("train_size_value", train_size_value)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   # sqrt = square-root 한국어로 제곱근. 결국 루트씌워서 계산한 값을 나타내 달라는 말.

def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
rmsle = RMSLE(y_test, y_predict)
print("MSE:", loss)
print("RMSE:", rmse)
print("RMSLE:", rmsle)


# 1.19 == epochs = 1000, batch_size = 381
# R2:  0.20423348554561715
# run time:  13.85 초
# random_state_value:  1
# train_size_value 0.71
# MSE: 1.3822555541992188
# RMSE: 164.20864812382456
# RMSLE: 1.175693721478541

# 1.18822 == epochs = 1000, batch_size = 381
# random_state_value:  1
# train_size_value 0.72



# 1.18143 == epochs = 1000, batch_size = 381
# R2:  0.19810355884237085
# run time:  14.59 초
# random_state_value:  1
# train_size_value 0.74
# MSE: 1.3773208856582642
# RMSE: 165.2521729059214
# RMSLE: 1.173593080617437



# epochs = 1000, batch_size = 323 / 정작 결과 안좋음. 1.2점대
# R2:  0.23197408937543884
# run time:  16.64 초
# random_state_value:  2
# train_size_value 0.74
# MSE: 1.366115927696228
# RMSE: 159.19413809083983
# RMSLE: 1.1688095686093125

# epochs = 1000, batch_size = 381
# R2:  0.22932758174089896
# run time:  14.5 초
# random_state_value:  2
# train_size_value 0.74
# MSE: 1.3315521478652954
# RMSE: 159.46818235012316
# RMSLE: 1.1539290919018346












