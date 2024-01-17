# https://dacon.io/competitions/open/235610/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, TextVectorization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time as tm

path = "c:/_data/dacon/wine/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")


train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']



# print(x)    # (5497, 12)
# print(y)    # (5497,)
# print(np.unique(y, return_counts=True))  
    # (array([3, 4, 5, 6, 7, 8, 9], dtype=int64),
    # array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
from sklearn.preprocessing import OneHotEncoder
y_ohe = y.values.reshape(-1, 1)
enc = OneHotEncoder(sparse=False).fit(y_ohe)
y_ohe = enc.transform(y_ohe)
print(y_ohe[0])
# print(np.unique(y, return_counts=True))  
# print(y_ohe)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, stratify = y, 
                                    train_size = 0.8, random_state = 0 )
# [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]   
# [0. 0. 1. 0. 0. 0. 0.]        


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


#2
# model = Sequential()
# model.add(Dense(80, input_dim = 12, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(7, activation = 'softmax'))
# # y의 label의 갯수 = 마지막 레이어숫자


# #3
# model.compile(loss= 'categorical_crossentropy', optimizer='adam',
#               metrics = ['acc'])
# es = EarlyStopping(monitor='val_loss', mode='auto',
#                    patience = 1000, verbose = 2,
#                    restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
#                       verbose=1,save_best_only=True,
#     filepath='c:/_data/_save/MCP/keras26_MCP_10_dacon_wine.hdf5')

# hist = model.fit(x_train, y_train, epochs= 20000,
#                  batch_size = 3517, validation_split= 0.2 ,
#                  verbose = 2, callbacks=[es, mcp])

model = load_model('c:/_data/_save/MCP//k26/10/k26_0117_1515_0703-1.0041.hdf5')

#4 
results = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
# encode 풀기
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
y_submit = np.argmax(y_submit, axis=1)+3

submission_csv['quality'] = y_submit
submission_csv.to_csv(path + "submission_0116_1.csv", index=False)

acc = accuracy_score(y_predict, y_test) 

print('loss:', results[0])
print('acc:', results[1])
print('accuracy_score :', acc)



# scaler = MinMaxScaler()
# loss: 1.1412980556488037
# acc: 0.5336363911628723
# accuracy_score : 0.5336363636363637


# scaler = StandardScaler()
# loss: 1.1396093368530273
# acc: 0.5318182110786438
# accuracy_score : 0.5318181818181819


# scaler = MaxAbsScaler()
# loss: 1.1051563024520874
# acc: 0.5363636612892151
# accuracy_score : 0.5363636363636364


# scaler = RobustScaler()
# loss: 1.1401103734970093
# acc: 0.5354545712471008


# MaxAbsScaler
# loss: 1.0724146366119385
# acc: 0.5336363911628723
# accuracy_score : 0.5336363636363637
