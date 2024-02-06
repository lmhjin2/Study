import numpy as np
import pandas as pd
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

datasets1 = datasets1.iloc[:1418]
datasets2 = datasets2.iloc[:1418]
###########################################################
datasets1 = datasets1.sort_values('일자', ascending=True)
datasets2 = datasets2.sort_values('일자', ascending=True)

x1 = datasets1.drop(['시가'], axis=1)
x1 = x1.drop(['종가'], axis=1)
x2 = datasets2.drop(['시가'], axis=1)
x2 = x2.drop(['종가'], axis=1)

y1 = datasets1['시가']
y2 = datasets2['종가']

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

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, y1, y2, 
    random_state = 981013 , 
    train_size = 0.9 , 
    shuffle=False
)

x1_train = x1_train.reshape(1266,80)
x1_test = x1_test.reshape(141,80)
x2_train = x2_train.reshape(1266,80)
x2_test = x2_test.reshape(141,80)
x1_predict = x1_predict.reshape(10,80)
x2_predict = x2_predict.reshape(10,80)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = StandardScaler()
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

model = load_model(path + 'sihum_0206_1308_0272-6979.07.hdf5')

loss = model.evaluate([x1_test, x2_test],[y1_test, y2_test])

y_predict =np.round(model.predict([x1_predict, x2_predict]),2)

ss = y_predict[0]
am = y_predict[1]

print('loss:',loss)
print('7일 삼성전자 시가:', ss[-1:])
print('7일 아모레 종가:', am[-1:])
print('합계:', ss[-1:] + am[-1:])
