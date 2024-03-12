import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

#1
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

random_state_value = 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, stratify=y, random_state = random_state_value )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
model = Sequential()
model.add(Dense(26, input_dim = 30))    # 364
model.add(Dense(52))                    # 1404
model.add(Dense(25))                    # 1325
model.add(Dense(50))                    # 1300
model.add(Dense(13))                    # 663
model.add(Dense(1, activation='sigmoid')) # 14

#3
from keras.optimizers import Adam, Adamax
# es = EarlyStopping(monitor='val_loss', mode = 'auto',
#                    patience = 20, verbose = 1, restore_best_weights = True )
learning_rate = 0.0001
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate))
hist = model.fit(x_train, y_train,
        #   callbacks=[es],
          validation_split = 0.2, epochs = 200, batch_size = 32 )
#4
loss = model.evaluate(x_test, y_test, verbose = 0)
print("lr : {0}, 로스 : {1}".format(learning_rate, loss))

y_predict = model.predict(x_test, verbose = 0)
# y_predict = (y_predict > 0.5).astype(int)
y_predict = np.argmax(y_predict, axis=1)
acc = accuracy_score(y_test, y_predict)
print("lr : {0}, ACC : {1}".format(learning_rate, acc))

# Epochs = 100
# lr : 1.0, 로스 : 60242368.0
# lr : 1.0, ACC : 0.3684210526315789
# lr : 0.1, 로스 : 31.135339736938477
# lr : 0.1, ACC : 0.3684210526315789
# lr : 0.01, 로스 : 0.02586577646434307
# lr : 0.01, ACC : 0.3684210526315789
# lr : 0.001, 로스 : 0.026218930259346962
# lr : 0.001, ACC : 0.3684210526315789
# lr : 0.0001, 로스 : 0.04181350767612457
# lr : 0.0001, ACC : 0.3684210526315789

# Epochs = 200
# lr : 1.0, 로스 : 6611304.0
# lr : 1.0, ACC : 0.3684210526315789
# lr : 0.1, 로스 : 1.8925707340240479
# lr : 0.1, ACC : 0.3684210526315789
# lr : 0.01, 로스 : 0.020893005654215813
# lr : 0.01, ACC : 0.3684210526315789
# lr : 0.001, 로스 : 0.02115112729370594
# lr : 0.001, ACC : 0.3684210526315789
# lr : 0.0001, 로스 : 0.03807621821761131
# lr : 0.0001, ACC : 0.3684210526315789