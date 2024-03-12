import numpy as np
import pandas as pd
from keras.optimizers import *
path = 'c:/_data/kaggle/Obesity_Risk/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder

lae_G = LabelEncoder()
train_csv['Gender'] = lae_G.fit_transform(train_csv['Gender'])
test_csv['Gender'] = lae_G.transform(test_csv['Gender'])

lae_fhwo = LabelEncoder()
train_csv['family_history_with_overweight'] = lae_fhwo.fit_transform(train_csv['family_history_with_overweight'])
test_csv['family_history_with_overweight'] = lae_fhwo.transform(test_csv['family_history_with_overweight'])

lae_FAVC = LabelEncoder()
train_csv['FAVC'] = lae_FAVC.fit_transform(train_csv['FAVC'])
test_csv['FAVC'] = lae_FAVC.transform(test_csv['FAVC'])

lae_CAEC = LabelEncoder()
train_csv['CAEC'] = lae_CAEC.fit_transform(train_csv['CAEC'])
test_csv['CAEC'] = lae_CAEC.transform(test_csv['CAEC'])

lae_SMOKE = LabelEncoder()
train_csv['SMOKE'] = lae_SMOKE.fit_transform(train_csv['SMOKE'])
test_csv['SMOKE'] = lae_SMOKE.transform(test_csv['SMOKE'])

lae_SCC = LabelEncoder()
train_csv['SCC'] = lae_SCC.fit_transform(train_csv['SCC'])
test_csv['SCC'] = lae_SCC.fit_transform(test_csv['SCC'])

lae_CALC = LabelEncoder()
test_csv['CALC'] = lae_CALC.fit_transform(test_csv['CALC'])
train_csv['CALC'] = lae_CALC.transform(train_csv['CALC'])

lae_MTRANS = LabelEncoder()
train_csv['MTRANS'] = lae_MTRANS.fit_transform(train_csv['MTRANS'])
test_csv['MTRANS'] = lae_MTRANS.transform(test_csv['MTRANS'])

lae_NObeyesdad = LabelEncoder()
train_csv['NObeyesdad'] = lae_NObeyesdad.fit_transform(train_csv['NObeyesdad'])

x = train_csv.drop(['NObeyesdad'], axis = 1)
y = train_csv['NObeyesdad']

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

y = np.array(y.values.reshape(-1,1))
y_ohe = OneHotEncoder(sparse=False).fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, stratify=y, test_size=0.2, random_state= 2 )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)
# print(x_train.shape, y_train.shape)     # (16606, 16) (16606, 7)
# print(np.unique(y_train, return_counts=True))
# print(np.unique(train_csv, return_counts=True))
from sklearn.metrics import accuracy_score, r2_score
from keras.models import *
from keras.layers import *

#2
model = Sequential()
model.add(Dense(172, input_shape = (16,)))
model.add(Dense(83,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(72, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(37, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3
learning_rate = 0.001
from keras.callbacks import ReduceLROnPlateau
rlr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, mode='auto')
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))
model.fit(x_train, y_train, epochs=200, batch_size=1000, verbose=1, callbacks=[rlr])
#4
results = model.evaluate(x_test, y_test, verbose=2)
y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)

y_test = np.argmax(y_test, axis = 1)            # argmax주석하면 에러
y_predict = np.argmax(y_predict, axis =1)       # argmax주석하면 에러
y_submit = np.argmax(y_submit, axis=1)          # argmax주석하면 에러
y_submit = lae_NObeyesdad.inverse_transform(y_submit)   # 주석하면 0점.

acc = accuracy_score(y_test, y_predict)
submission_csv['NObeyesdad'] = y_submit
submission_csv.to_csv(path + "submission_0312_3.csv", index=False)

print("lr : {0}, ACC : {1}, loss : {2}".format(learning_rate, acc, results))
# https://www.kaggle.com/c/playground-series-s4e2/overview

# results: 0.8882466281310212
# acc: 0.8882466281310212
# tts = 5

# Epochs = 100
# lr : 1.0, ACC : 0.19484585741811175, loss : 1.932090163230896
# lr : 0.1, ACC : 0.7996146435452793, loss : 0.5947147607803345
# lr : 0.01, ACC : 0.867533718689788, loss : 0.39873191714286804
# lr : 0.001, ACC : 0.8704238921001927, loss : 0.37442535161972046
# lr : 0.0001, ACC : 0.8728323699421965, loss : 0.364547997713089


# Epochs = 200
# lr : 1.0, ACC : 0.19484585741811175, loss : 1.937357783317566
# lr : 0.1, ACC : 0.8588631984585742, loss : 0.43148142099380493
# lr : 0.01, ACC : 0.8740366088631984, loss : 0.39246636629104614
# lr : 0.001, ACC : 0.878131021194605, loss : 0.3682706356048584
# lr : 0.0001, ACC : 0.8817437379576107, loss : 0.3555486500263214

# lr : 0.001, ACC : 0.8759633911368016, loss : 0.38026636838912964