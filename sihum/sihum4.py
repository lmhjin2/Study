import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Conv2D, SimpleRNN, LSTM, GRU, Dropout, Flatten, Embedding, Reshape, Input, concatenate, Concatenate, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# csv 불러오기 + 컬럼명 지정
path = 'c:/_data/sihum/'
samsung_csv = pd.read_csv(path + '삼성 240205.csv', encoding='euc-kr', index_col=0, header=0, names = ['일자','시가','고가','저가','종가','전일비','등락폭','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비'])
amore_csv = pd.read_csv(path + '아모레 240205.csv', encoding='euc-kr', index_col=0, header=0, names = ['일자','시가','고가','저가','종가','전일비','등락폭','등락률','거래량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램','외인비'])

    # print(samsung_csv)    # (10296, 16)
    # print(amore_csv)      # (4350, 16)
    # print(samsung_csv.dtypes)

# 전일비 원핫인코딩     /    상승 = 2 , 동결 = 1, 하락 = 0
samsung_csv['전일비'] = samsung_csv['전일비'].str.replace('▲','2').str.replace('↑','2').str.replace('▼','0').str.replace('↓','0').str.replace(' ','1')
samsung_csv['전일비'] = samsung_csv['전일비'].astype('float64')
    # print(np.unique(samsung_csv['전일비'], return_counts=True))
amore_csv['전일비'] = amore_csv['전일비'].str.replace('▲','2').str.replace('↑','2').str.replace('▼','0').str.replace('↓','0').str.replace(' ','1')
amore_csv['전일비'] = amore_csv['전일비'].astype('float64')
    # print(np.unique(amore_csv['전일비'], return_counts=True))

# object -> float64
for col in samsung_csv.columns:
    if samsung_csv[col].dtype != 'float64':
        samsung_csv[col] = pd.to_numeric(samsung_csv[col].str.replace(',',''), errors='coerce')
samsung_csv = samsung_csv.astype('float64')
    # print(samsung_csv.dtypes)
for col in amore_csv.columns:
    if amore_csv[col].dtype != 'float64':
        amore_csv[col] = pd.to_numeric(amore_csv[col].str.replace(',',''), errors='coerce')
amore_csv = amore_csv.astype('float64')
    # print(amore_csv.dtypes)
    # print(samsung_csv['전일비'])

# 옛날데이터부터 접근하게끔 순서 변경
samsung_csv = samsung_csv.sort_values(['일자'], ascending=[True])
amore_csv = amore_csv.sort_values(['일자'], ascending=[True])

    # print(samsung_csv['외인비'])
    # print(samsung_csv.dtypes)
    # print(amore_csv.columns)
    # print(amore_csv)
    # print(samsung_csv.head())
# # 전일비 드랍
# samsung_csv.drop('전일비', axis=1, inplace=True)
# amore_csv.drop('전일비', axis=1, inplace=True)

# '시가' 컬럼 맨 오른쪽으로 보내기
timeprice = '시가'
moved_column1 = samsung_csv.pop(timeprice)
samsung_csv[timeprice] = moved_column1

endprice = '종가'
moved_column2 = amore_csv.pop(endprice)
amore_csv[endprice] = moved_column2

# print(samsung_csv.columns) # 확인 완

# 액면분할 하기 전 데이터에 직빵으로 나누고 곱하기. (액면분할 이후와 동일하게끔)
samsung_csv_reset = samsung_csv.reset_index()
samsung_before= samsung_csv_reset[samsung_csv_reset['일자'] <= '2018/05/03']
samsung_after = samsung_csv_reset[samsung_csv_reset['일자'] > '2018/05/03']

amore_csv_reset = amore_csv.reset_index()
amore_before= amore_csv_reset[amore_csv_reset['일자'] <= '2015/05/07']
amore_after = amore_csv_reset[amore_csv_reset['일자'] > '2015/05/07']

samsung_before.loc[:,['시가','고가','저가','종가','등락폭','금액(백만)']] = samsung_before.loc[:,['시가','고가','저가','종가','등락폭','금액(백만)']] / 50
samsung_before.loc[:,['거래량','개인','기관','외인(수량)','외국계','프로그램']] = samsung_before.loc[:,['거래량','개인','기관','외인(수량)','외국계','프로그램']] * 50

amore_before.loc[:,['시가','고가','저가','종가','등락폭','금액(백만)']] = amore_before.loc[:,['시가','고가','저가','종가','등락폭','금액(백만)']] / 10
amore_before.loc[:,['거래량','개인','기관','외인(수량)','외국계','프로그램']] = amore_before.loc[:,['거래량','개인','기관','외인(수량)','외국계','프로그램']] * 10

# print(amore_before)

samsung_concat = pd.concat([samsung_before, samsung_after])
samsung_concat = samsung_concat[samsung_concat['일자'] > '2006/06/28']  # amore랑 데이터 수 맞춰주려고 이전 데이터 날림
samsung = samsung_concat.set_index('일자')
# print(samsung)

amore_concat = pd.concat([amore_before, amore_after])
amore = amore_concat.set_index('일자')
# print(amore)

# 결측치에 0 넣기
samsung.fillna(0, inplace=True)
amore.fillna(0, inplace=True)
# print(samsung)
# print(amore)
# print(samsung.shape)
# samsung_max = samsung.max().max()
# amore_max = amore.max().max()   #  3억2662만2000
# print(samsung_max, amore_max)   # (326622000, 3896837)

# y값 지정.
def split_x(dataset, timestep, column):
    x_arr = []
    y_arr = []
    for i in range(len(dataset) - timestep - 1):
        x_subset = dataset.iloc[i:(i+timestep), :]
        y_subset = dataset.iloc[(i+timestep+1), column]
        x_arr.append(x_subset)
        y_arr.append(y_subset)
    return np.array(x_arr), np.array(y_arr)

# print(samsung.iloc[:,15:]) = print(samsung['시가'])

samsung_x, samsung_y = split_x(samsung, 20, 15)
amore_x, amore_y = split_x(amore, 20, 15)
# print(samsung_x)
# print(amore_x)
# print(samsung_x.shape)

# predict_samsung

# 스케일링
samsung_x = samsung_x.reshape(4329*20, 16)
amore_x = amore_x.reshape(4329*20 ,16)
scaler = RobustScaler()
scaler.fit(samsung_x)
# print(samsung_x.shape)  # (4329, 20, 16)
samsung_x = scaler.transform(samsung_x)
amore_x = scaler.transform(amore_x)

samsung_x = samsung_x.reshape(-1,20,16)
amore_x = amore_x.reshape(-1,20,16)




#######################################태홍  predict 데이터 만들기 ##############################################
# predict_samsung = samsung_concat[samsung_concat['일자'] > '2024/01/08']
# predict_samsung = predict_samsung.set_index('일자')
#     # print(predict_samsung.shape)

# predict_amore = amore_concat[amore_concat['일자'] > '2024/01/08']
# predict_amore = predict_amore.set_index('일자')

    # print(predict_amore.shape)

# 최종 predict 데이터 스케일링
# predict_samsung = pd.DataFrame(columns = predict_samsung.columns, index=predict_samsung.index)
# predict_amore = pd.DataFrame(columns=predict_amore.columns, index = predict_amore.index)

# predict_samsung = predict_samsung.values.reshape(1,20,16)
# predict_amore = predict_amore.values.reshape(1,20,16)
# print(predict_samsung.dtype)
# print(samsung_x.dtype)


predict_samsung = samsung.to_numpy()
predict_amore = amore.to_numpy()

predict_samsung = predict_samsung[-20:]
predict_amore = predict_amore[-20:]


# print(predict_samsung)
# print(predict_samsung.shape)
# print(predict_amore)
predict_samsung = predict_samsung.reshape(1,20,16)
predict_amore = predict_amore.reshape(1,20,16)

###################################################################################################################
# train_test_val_split
samsung_x_train, samsung_x_split, samsung_y_train, samsung_y_split = train_test_split(samsung_x, samsung_y, test_size=0.4, random_state= 0 , shuffle=False)
samsung_x_val, samsung_x_test, samsung_y_val, samsung_y_test = train_test_split(samsung_x_split, samsung_y_split, test_size=0.5, random_state= 0 , shuffle=False)

amore_x_train, amore_x_split, amore_y_train, amore_y_split = train_test_split(amore_x, amore_y, test_size=0.4, random_state= 0 , shuffle=False)
amore_x_val, amore_x_test, amore_y_val, amore_y_test = train_test_split(amore_x_split, amore_y_split, test_size=0.5, random_state= 0 , shuffle=False)

# print(samsung_x_train.shape, amore_x_train.shape)   # (2597, 20, 16) (2597, 20, 16)
# print(samsung_x_test.shape, amore_x_test.shape)     # (866, 20, 16) (866, 20, 16)
# print(samsung_x_val.shape, amore_x_val.shape)       # (866, 20, 16) (866, 20, 16)

# amore = 4351
# samsung = 2006/06/29 부터 현재가 4351 // 처음부터 2006/06/28 까지는 날림. 10297개중에 5946개를 날림. 이게 맞냐?

# 2-1 ss
input_ss = Input(shape=(20,16))
dense1 = Bidirectional(LSTM(32, return_sequences=True, name='ss1'))(input_ss)
dense2 = LSTM(64, name='ss2')(dense1)
dense3 = Dense(128, name='ss3')(dense2)
dense4 = Dense(32, name='ss4')(dense3)
dense5 = Dense(16, name='ss5')(dense4)
output_ss = Dense(1, activation='linear', name='ssout')(dense5)

# 2-2 am
input_am = Input(shape=(20,16))
dense11 = Bidirectional(LSTM(32, return_sequences=True, name='am1'))(input_am)
dense12 = LSTM(64, name='am2')(dense11)
dense13 = Dense(128, name='am3')(dense12)
dense14 = Dense(32, name='am4')(dense13)
dense15 = Dense(16, name='am5')(dense14)
output_am = Dense(1, activation='linear', name='amout')(dense15)

# 2-3 concatenate
merge1 = concatenate([output_ss, output_am], name='mg1')
merge2 = Dense(64, name='mg2')(merge1)
merge3 = Dense(32, name = 'mg3')(merge2)
merge4 = Dense(16, name = 'mg4')(merge3)
last_ss = Dense(1, name='ss_last')(merge4)
last_am = Dense(1, name='am_last')(merge4)

model = Model(inputs = [input_ss, input_am], outputs = [last_ss,last_am])

model.summary()

#3
model.compile(loss='mae', optimizer='adam', loss_weights=[1.0, 1.0])
es = EarlyStopping(monitor='val_loss', mode='auto', patience= 50 , verbose=1, restore_best_weights=True)
model.fit([samsung_x_train, amore_x_train], [samsung_y_train, amore_y_train], epochs= 100 , batch_size= 32 , verbose=1, callbacks=[es],
          validation_data=([samsung_x_val, amore_x_val], [samsung_y_val, amore_y_val]))

# predict_y = model.predict([predict_samsung, predict_amore]) # 최종 predict

# print(samsung_x_test.shape)

y_predict = model.predict([samsung_x_test, amore_x_test])

test_ss = y_predict[0]
test_am = y_predict[1]

# print(test_ss.shape, test_am.shape)

# print(samsung_x_test.shape, test_ss.shape)  # (866, 20, 16) (866, 20, 1)

    # samsung_x_test = samsung_x_test.reshape(866, 20*16)
    # amore_x_test = amore_x_test.reshape(866, 20*16)

# test_ss = test_ss.reshape(866, 20*16)
# test_am = test_am.reshape(866, 20*16)

#4
loss = model.evaluate([samsung_x_test, amore_x_test], [samsung_y_test, amore_y_test])

# print(samsung_y_test.shape, test_ss.shape)   # (866,) (1,)

# r2_ss = r2_score(samsung_y_test, test_ss)
# r2_am = r2_score(amore_y_test, test_am)
# print('r2:', r2_ss, r2_am)
# print('test_ss', test_ss)
# print('test_am', test_am)

# print('test_ss', test_ss.shape)
# print('test_am', test_am.shape)
print('loss:', loss)
# test_ss = np.round(test_ss,2)
# print('7일 삼성전자 시가:', test_ss[-1:])
# test_am = np.round(test_am,2)
# print('7일 아모레 종가:', test_am[-1:])


pred_sihum = model.predict([predict_samsung, predict_amore])
sihum_ss = pred_sihum[0]
sihum_am = pred_sihum[1]

sihum_ss = np.round(sihum_ss,2)
print('7일 삼성전자 시가:', sihum_ss[-1:])
sihum_am = np.round(sihum_am,2)
print('7일 아모레 종가:', sihum_am[-1:])

