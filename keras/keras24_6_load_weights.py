# 09_1 copy
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import time

#1
datasets = load_boston()

x = datasets.data
y = datasets.target

random_state_value = 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.3, random_state = random_state_value )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

#2
model = Sequential()
model.add(Dense(26, input_dim = 13))    # 364
model.add(Dense(52))                    # 1404
model.add(Dense(25))                    # 1325
model.add(Dense(50))                    # 1300
model.add(Dense(13))                    # 663
model.add(Dense(1))                     # 14

# model.summary()

# model.save("c:/_data/_save/keras24_save_model.h5")  # c드라이브 부터 들어가는게 # 절대경로
# model.save("../_data/_save/keras24_save_model.h5")  ../은 상위폴더를 의미 # 이건 상대경로
# model.save("./_data/_save/keras24_save_model.h5")   ./은 현재 Study폴더에 _data폴더, 그 안에 _save폴더 그안에 h5 파일이 생김.

# model.save_weights('c:/_data/_save/keras24_5_save_weights1.h5')
model.load_weights('c:/_data/_save/keras24_5_save_weights2.h5')

# model = load_model('../_data/_save/keras24_3_save_model2.h5')
model.summary()

#3
model.compile(loss='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs = 100, batch_size = 30 )

# model.save_weights('c:/_data/_save/keras24_5_save_weights2.h5')  # 훈련된 모델


#4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("로스 :", loss)
print("R2 스코어 :", r2)
print("Random State:", random_state_value)
print("boston")

import warnings
warnings.filterwarnings('ignore')

# 0.7845982031606218

# random_state = 1, epochs = 8000, batch_size = 30, loss = 'mse'
# 로스 : 18.694143295288086
# R2 스코어 : 0.796036779999476
# 걸린시간 :  58.3 초

# scaler = MinMaxScaler()
# 로스 : 3.1486480236053467
# R2 스코어 : 0.7835180345040087
# 걸린시간 :  9.9 초

# scaler = StandardScaler()
# 로스 : 3.144235610961914
# R2 스코어 : 0.7939326089286648
# 걸린시간 :  9.63 초

# scaler = MaxAbsScaler()
# 로스 : 3.134629487991333
# R2 스코어 : 0.7852473736299466
# 걸린시간 :  9.71 초

# scaler = RobustScaler()
# 로스 : 3.1448447704315186
# R2 스코어 : 0.7995639930028042
# 걸린시간 :  9.69 초


# RoubustScaler
