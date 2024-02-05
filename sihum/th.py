import numpy as np
import pandas as pd

path = 'c:/_data/sihum/'
path_save = 'c:/_data/sihum/'

datasets1 = pd.read_csv(path + '삼성 240205.csv', index_col = 0, encoding = 'euc-kr', thousands = ',')
datasets2 = pd.read_csv(path + '아모레 240205.csv', index_col =0, encoding = 'euc-kr', thousands = ',')

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

x1 = datasets1.drop(['시가'], axis=1)
x1 = datasets1.drop(['종가'], axis=1)
x2 = datasets2.drop(['시가'], axis=1)
x2 = datasets2.drop(['시가'], axis=1)
y1 = datasets1['시가']
y2 = datasets2['종가']

# print(x1.isnull().sum())
# print(x2.isnull().sum())
# print(x1.shape) # (1418, 9)


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


from sklearn.model_selection import train_test_split
# x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1,x2,y1,y2, random_state = 158796, train_size = 0.9, shuffle=False)
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, y1, y2, 
    random_state=158796, 
    train_size=0.9, 
    shuffle=False
)
# print(x1_train.shape, x1_test.shape) #(1266, 10, 9) (141, 10, 9)
# print(x2_train.shape, x2_test.shape) #(1266, 10, 9) (141, 10, 9)
# print(x1_predict.shape, x2_predict.shape) #(10, 10, 9) (10, 10, 9)

x1_train = x1_train.reshape(1266,90)
x1_test = x1_test.reshape(141,90)
x2_train = x2_train.reshape(1266,90)
x2_test = x2_test.reshape(141,90)
x1_predict = x1_predict.reshape(10,90)
x2_predict = x2_predict.reshape(10,90)

from sklearn.preprocessing import MinMaxScaler, RobustScaler
scaler = MinMaxScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)

x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)

x1_predict = scaler.transform(x1_predict)
x2_predict = scaler.transform(x2_predict)


x1_train = x1_train.reshape(1266,10,9)
x1_test = x1_test.reshape(141,10,9)
x2_train = x2_train.reshape(1266,10,9)
x2_test = x2_test.reshape(141,10,9)
x1_predict = x1_predict.reshape(10,10,9)
x2_predict = x2_predict.reshape(10,10,9)



#모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, Bidirectional, Flatten, Dropout, Dense
#모델1
input1 = Input(shape = (10,9))
lstm1  = Bidirectional(LSTM(32, return_sequences=True))(input1)
lstm2 = Bidirectional(LSTM(64, return_sequences=True))(lstm1)
lstm3 = LSTM(32)(lstm2)
dense1 = Dense(64)(lstm3)
dense2 = Dense(32)(dense1)
dense3 = Dense(16)(dense2)
output1 = Dense(8)(dense3)
#모델2
input2 = Input(shape = (10,9))
lstm11 = LSTM(32, return_sequences=True)(input2)
lstm12 = Bidirectional(LSTM(64, return_sequences=True))(lstm11)
lstm13 = Bidirectional(LSTM(32))(lstm12)
dense11 = Dense(64)(lstm13)
dense14 = Dense(32)(dense11)
output2 = Dense(16)(dense14)
#머지
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])
out1 = Dense(32)(merge1)
out1 = Dense(16)(out1)
out1 = Dense(1)(out1)

out2 = Dense(32)(merge1)
out2 = Dense(64)(out2)
out2 = Dense(1)(out2)

model = Model(inputs = [input1,input2], outputs = [out1, out2])
model.summary()

#컴파일 훈련
model.compile(loss='mae', optimizer='adamax')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = 'c:/_data/sihum/'
filename = '{epoch:04d}-{val_loss:.2f}.hdf5'

import datetime
date = datetime.datetime.now() 
date = date.strftime("%m%d_%H%M")

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 150, restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'min',verbose=1, save_best_only=True, filepath ="".join([filepath,'시험_',date,'_',filename]))
model.fit([x1_train, x2_train],[y1_train, y2_train], epochs=10000, batch_size=16, validation_split=0.2, callbacks=[es,mcp])

#평가 예측
loss = model.evaluate([x1_test, x2_test],[y1_test, y2_test])
print('loss:',loss)



y_predict =np.round(model.predict([x1_predict, x2_predict]),2)

ss = y_predict[0]
am = y_predict[1]

print('7일 삼성전자 시가:', ss[-1:])
print('7일 아모레 종가:', am[-1:])
#74200
#1204006

#0137_0091 
#7일 삼성전자 시가: [[69425.64]]
# 7일 아모레 종가: [[137076.58]]

#0143_0180
# 7일 삼성전자 시가: [[74839.96]]
# 7일 아모레 종가: [[146881.56]]



