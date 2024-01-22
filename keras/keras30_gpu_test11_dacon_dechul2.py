# https://dacon.io/competitions/official/236214/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time as tm

path = "c:/_data/dacon/dechul/"
train_csv = pd.read_csv(path + 'train.csv', index_col=0)  
test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
submission_csv = pd.read_csv(path + 'sample_submission.csv')


le_work_period = LabelEncoder() 
le_work_period.fit(train_csv['근로기간'])
train_csv['근로기간'] = le_work_period.transform(train_csv['근로기간'])
test_csv['근로기간'] = le_work_period.transform(test_csv['근로기간'])

le_grade = LabelEncoder()
le_grade.fit(train_csv['대출등급'])
train_csv['대출등급'] = le_grade.transform(train_csv['대출등급'])

le_purpose = LabelEncoder()
test_csv.iloc[34486,7] = '이사'     # 결혼 -> 이사 로 임의로 바꿈
le_purpose.fit(train_csv['대출목적'])
train_csv['대출목적'] = le_purpose.transform(train_csv['대출목적'])
test_csv['대출목적'] = le_purpose.transform(test_csv['대출목적'])

le_own = LabelEncoder()
le_own.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = le_own.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = le_own.transform(test_csv['주택소유상태'])

le_loan_period = LabelEncoder()
le_loan_period.fit(train_csv['대출기간'])
train_csv['대출기간'] = le_loan_period.transform(train_csv['대출기간'])
test_csv['대출기간'] = le_loan_period.transform(test_csv['대출기간'])


x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']





y = y.values.reshape(-1,1)
y_ohe = OneHotEncoder(sparse=False).fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y_ohe, stratify=y, test_size = 0.18, random_state = 1716 )   # 1717, 1818

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2
model = Sequential()
model.add(Dense(64, input_shape = (13,), activation = 'relu'))
model.add(Dense(48, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(48, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

# model = Sequential()
# model.add(Dense(64, input_shape = (13,), activation = 'relu'))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(7, activation = 'softmax'))

#3
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   # 월일_시분

path1 = "c:/_data/_save/MCP/k28/11/"
filename = "{epoch:04d}-{val_loss:.4f}.hdf5"
filepath = "".join([path1, 'k28_', date, '_1_', filename])

model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])

es = EarlyStopping(monitor='val_loss', mode = 'auto',
                   patience = 1500, verbose = 2,
                   restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto',
                      verbose=1, save_best_only=True,
    filepath=filepath)
start_time = tm.time()
hist = model.fit(x_train, y_train, epochs = 50000,
                 batch_size = 500, validation_split = 0.18,
                 verbose = 2, callbacks=[es, mcp] )
end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4
results = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis = 1)
y_predict = np.argmax(y_predict, axis =1)
y_submit = np.argmax(y_submit, axis=1)
y_submit = le_grade.inverse_transform(y_submit)

submission_csv['대출등급'] = y_submit
submission_csv.to_csv(path + "submission_0121_rbs1.csv", index=False)

acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average = 'macro') # [None, 'micro', 'macro', 'weighted'] 중에 하나

print('accuracy_score :', acc)
print('loss', results[0])
print('f1 score', f1)
print('run time', run_time)

# scaler = MinMaxScaler()
# accuracy_score : 0.9191715225292794
# run time 4188.69
# loss 0.5466686487197876
# f1 score 0.8978954857467664

# scaler = StandardScaler()
# accuracy_score : 0.8998442277736111
# run time 2353.3
# loss 0.3901255130767822
# f1 score 0.8632198023469781

# scaler = MaxAbsScaler()
# accuracy_score : 0.9096521086943979   
# run time 5581.98   
# loss 0.42253240942955017
# f1 score 0.8794265224077572

# scaler = RobustScaler()
# accuracy_score : 0.8962672359083829
# run time 1869.31
# loss 0.4001994729042053
# f1 score 0.8628113819593662



# CPU
# 94.08 초

# GPU
# 89.58 초

