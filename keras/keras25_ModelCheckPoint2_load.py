# 09_1 copy
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
# from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

#1
datasets = load_boston()

x = datasets.data
y = datasets.target

random_state_value = 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = random_state_value )

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
# model = Sequential()
# model.add(Dense(26, input_dim = 13))    # 364
# model.add(Dense(52))                    # 1404
# model.add(Dense(25))                    # 1325
# model.add(Dense(50))                    # 1300
# model.add(Dense(13))                    # 663
# model.add(Dense(1))                     # 14
# # model.summary()

#3
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode = 'auto',
#                    patience = 10, verbose = 1, restore_best_weights = True )
# mcp = ModelCheckpoint(
#     monitor='val_loss', mode = 'auto', verbose = 1, save_best_only=True,
#     filepath="c:/_data/_save/MCP/keras_25_MCP1.hdf5")

# model.compile(loss='mae', optimizer='adam')
# hist = model.fit(x_train, y_train,
#           callbacks = [es, mcp], validation_split = 0.2,
#           epochs = 1000, batch_size = 32 )

model = load_model('c:/_data/_save/MCP/keras25_MCP1.hdf5')
# ModelCheckpoint

#4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("로스 :", loss)
print("R2 스코어 :", r2)
# print("Random State:", random_state_value)
# print("boston")
print('=================================================================================')
# print(hist.history)
print('=================================================================================')
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
