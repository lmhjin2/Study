import numpy as np
import pandas as pd

path = 'c:/_data/sihum/'

datasets1 = pd.read_csv(path + '삼성 240205.csv', index_col = 0, encoding = 'euc-kr', thousands = ',')
datasets2 = pd.read_csv(path + '아모레 240205.csv', index_col = 0, encoding = 'euc-kr', thousands = ',')

# print(datasets1.head)

datasets1 = datasets1.drop(['전일비'], axis=1)
datasets2 = datasets2.drop(['전일비'], axis=1)
datasets1 = datasets1.drop(['Unnamed: 6'], axis=1)
datasets2 = datasets2.drop(['Unnamed: 6'],axis=1)
datasets1 = datasets1.drop(['외인(수량)'], axis=1)
datasets2 = datasets2.drop(['외인(수량)'], axis=1)
datasets1 = datasets1.drop(['외국계'], axis=1)
datasets2 = datasets2.drop(['외국계'], axis=1)
datasets1 = datasets1.drop(['프로그램'], axis=1)
datasets2 = datasets2.drop(['프로그램'], axis=1)
datasets1 = datasets1.drop(['기관'], axis=1)
datasets2 = datasets2.drop(['기관'], axis=1)

# print(type(datasets1)) #<class 'pandas.core.frame.DataFrame'>
# print(type(datasets2)) #<class 'pandas.core.frame.DataFrame'>
# print(datasets1.info())
# print(datasets2.info())

datasets1 = datasets1.iloc[:1418]
datasets2 = datasets2.iloc[:1418]
###########################################################
datasets1 = datasets1.sort_values('일자', ascending=True)
datasets2 = datasets2.sort_values('일자', ascending=True)

# print(datasets1.head)
# print(datasets1.shape)  # (1418, 10)

print(datasets2.dtypes)


x1 = datasets1.drop(['시가'], axis=1)
x1 = x1.drop(['종가'], axis=1)
x2 = datasets2.drop(['시가'], axis=1)
x2 = x2.drop(['종가'], axis=1)

y1 = datasets1['시가']
y2 = datasets2['종가']

# print(x1.isnull().sum())
# print(x2.isnull().sum())
# print(x1.shape) # (1418, 8)


# import matplotlib.pyplot as plt
# # plt.plot(x1['거래량']) #거래량 robust
# # plt.plot(x1['등락률']) #등락률 robust
# # plt.plot(x1['금액(백만)']) #금액 백만 robust
# # plt.plot(x1['개인']) #개인 robust
# # plt.plot(x1['기관']) #기관 robust
# plt.plot(x1['외인비']) #외인(수량) robust 외국계 robust 프로그램
                  
# plt.show()
##########################################################
timesteps = 10
def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps -1):  
        subset = dataset[i : (i+timesteps)] 
        aaa.append(subset)      
    return np.array(aaa)

x1 = split_x(x1, timesteps)
x2 = split_x(x2, timesteps)

y1 = y1[timesteps+1:]
y2 = y2[timesteps+1:]

x1_predict = x1[-10:]
x2_predict = x2[-10:]
#########################################################
# print(x2[-1])

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, y1, y2, 
    random_state = 112233 , 
    train_size = 0.9 , 
    shuffle=False
)
# print(x1_train.shape, x1_test.shape) #(1266, 10, 9) (141, 10, 9)
# print(x2_train.shape, x2_test.shape) #(1266, 10, 9) (141, 10, 9)
# print(x1_predict.shape, x2_predict.shape) #(10, 10, 9) (10, 10, 9)

x1_train = x1_train.reshape(1266,80)
x1_test = x1_test.reshape(141,80)
x2_train = x2_train.reshape(1266,80)
x2_test = x2_test.reshape(141,80)
x1_predict = x1_predict.reshape(10,80)
x2_predict = x2_predict.reshape(10,80)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)

x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)

x1_predict = scaler.transform(x1_predict)
x2_predict = scaler.transform(x2_predict)

x1_train = x1_train.reshape(1266,10,8)
x1_test = x1_test.reshape(141,10,8)
x2_train = x2_train.reshape(1266,10,8)
x2_test = x2_test.reshape(141,10,8)
x1_predict = x1_predict.reshape(10,10,8)
x2_predict = x2_predict.reshape(10,10,8)


#모델구성
from keras.models import Model
from keras.layers import LSTM, Input, Bidirectional, Flatten, Dropout, Dense
#모델1
input1 = Input(shape = (10,8))
lstm1  = Bidirectional(LSTM(32, return_sequences=True, name = 'ls1'))(input1)
lstm2 = Bidirectional(LSTM(64, return_sequences=True, name='ls2'))(lstm1)
lstm3 = LSTM(32, name='ls3')(lstm2)
dense1 = Dense(64, name='d1')(lstm3)
dense2 = Dense(32, name='d2')(dense1)
dense3 = Dense(16, name='d3')(dense2)
output1 = Dense(8, name='o1')(dense3)

#모델2
input2 = Input(shape = (10,8))
lstm11 = LSTM(32, return_sequences=True, name='ls11')(input2)
lstm12 = Bidirectional(LSTM(64, return_sequences=True, name='ls12'))(lstm11)
lstm13 = Bidirectional(LSTM(32, name='ls13'))(lstm12)
dense11 = Dense(64, name='d11')(lstm13)
dense12 = Dense(32, name='d12')(dense11)
output2 = Dense(16, name='o2')(dense12)
#머지
from keras.layers import concatenate
merge1 = concatenate([output1, output2])
out1 = Dense(32, name='out1')(merge1)
out2 = Dense(16, name='out2')(out1)
out3 = Dense(1, name = 'ss')(out2)

out21 = Dense(32, name='out21')(merge1)
out22 = Dense(64, name='out22')(out21)
out23 = Dense(1, name = 'am')(out22)

model = Model(inputs = [input1,input2], outputs = [out3, out23])
model.summary()

#컴파일 훈련
model.compile(loss='mae', optimizer='adamax', loss_weights=[1.0, 1.0])
from keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = 'c:/_data/sihum/save/'
filename = '{epoch:04d}-{val_loss:.2f}.hdf5'

import datetime
date = datetime.datetime.now() 
date = date.strftime("%m%d_%H%M")

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 200 , restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'min',verbose=1, 
                      save_best_only=True, filepath ="".join([filepath,'sihum_',date,'_',filename]))
model.fit([x1_train, x2_train],[y1_train, y2_train], 
          epochs=100000 , batch_size = 33 , validation_split= 0.2 , 
          callbacks=[es,mcp])

#평가 예측
loss = model.evaluate([x1_test, x2_test],[y1_test, y2_test])
print('loss:',loss)

y_predict =np.round(model.predict([x1_predict, x2_predict]),2)

ss = y_predict[0]
am = y_predict[1]

print('7일 삼성전자 시가:', ss[-1:])
print('7일 아모레 종가:', am[-1:])

# 74200
# 125300

# sihum_0206_1105_0043-7108.93
# 7일 삼성전자 시가: [[75180.61]]
# 7일 아모레 종가: [[139686.98]]

# sihum_0206_1207_0133-5182.92
# 7일 삼성전자 시가: [[74531.3]]
# 7일 아모레 종가: [[131342.73]]
