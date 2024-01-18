# https://dacon.io/competitions/open/236068/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1 
path = "c:/_data/dacon/diabetes/"
train_csv = pd.read_csv(path + "train.csv", index_col = 0)
# (652, 9)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
# (116, 8)
submission_scv = pd.read_csv(path + "sample_submission.csv")

x = train_csv.drop(['Outcome'], axis = 1)   # (652, 8)
y = train_csv['Outcome']    # (652,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state = 1 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2
# model = Sequential()
# model.add(Dense(120, input_dim = 8))
# model.add(Dropout(0.2))
# model.add(Dense(90))
# model.add(Dense(60))
# model.add(Dropout(0.2))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(1, activation='sigmoid')) # 0에서 1사이의 값으로 내기 위함


#2
input1 = Input(shape=(8,))
dense1 = Dense(120)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(90)(drop1)
dense3 = Dense(60)(dense2)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(30)(drop2)
dense5 = Dense(10)(dense4)
output1 = Dense(1, activation='sigmoid')(dense5)

model = Model(inputs = input1, outputs = output1)


#3
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   # 월일_시분

path1 = "c:/_data/_save/MCP/k28/07/"
filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
filepath = "".join([path1, 'k28_', date, '_', filename])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience = 500, verbose = 1,
                   restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',
                      verbose=1,save_best_only=True,
    filepath=filepath)
hist = model.fit(x_train, y_train, epochs = 3000,
                 batch_size = 25, validation_split = 0.13,
                 verbose = 3, callbacks = [es, mcp])

#4
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)

submission_scv['Outcome'] = np.round(y_submit)
submission_scv.to_csv(path + "submission_0117_1.csv", index = False)

def ACC(y_test, y_predict):
    return accuracy_score(y_test, y_predict)

acc = ACC(y_test, np.round(y_predict))
print('loss', loss)
print('acc:', acc)


# 점수 : 0.775862069   batch 25, random 1, monitor val_loss, patience = 500
# loss [0.3650536835193634, 0.8367347121238708]
# acc: 0.8367346938775511

# scaler = MinMaxScaler()
# loss [0.3738122582435608, 0.8673469424247742] 
# acc: 0.8673469387755102

# scaler = StandardScaler()
# loss [0.3663322329521179, 0.8775510191917419] 
# acc: 0.8775510204081632

# scaler = MaxAbsScaler()
# loss [0.3781272768974304, 0.8571428656578064] 
# acc: 0.8571428571428571

# scaler = RobustScaler()
# loss [0.36750856041908264, 0.8673469424247742]
# acc: 0.8673469387755102


# 안쓰는게 낫다?