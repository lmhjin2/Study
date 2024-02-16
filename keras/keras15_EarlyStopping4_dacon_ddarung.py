# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error,accuracy_score

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
path = "c:/_data\\dacon/ddarung//"
train_csv = pd.read_csv(path+"train.csv", index_col = 0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())
x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
<<<<<<< HEAD
    x, y, train_size = 0.85, random_state = 151717 )

#2
model = Sequential()
model.add(Dense(828, input_dim = 9))
model.add(Dense(560))
model.add(Dense(330))
model.add(Dense(180))
model.add(Dense(108))
model.add(Dense(52))
model.add(Dense(27))
=======
    x, y, test_size = 0.85, random_state = 1234 )

#2
model = Sequential()
model.add(Dense(250, input_dim = 9, activation = 'relu'))
model.add(Dense(180, activation = 'relu'))
model.add(Dense(110, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(70, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
>>>>>>> faa0e1cfdd5fe7c1df8683dad05a659a5e99e189
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
<<<<<<< HEAD
                   patience = 200,
                   verbose = 1,
                   restore_best_weights=True)
# restore   69.7278194041
# loss: 2624.61181640625
# r2: 0.558611087447247
# restore   70.4002509923
# loss: 2566.83056640625
# r2: 0.6217188221750047
hist = model.fit(x_train, y_train, epochs = 3000,
                 batch_size = 109,
          validation_split = 0.13, verbose = 2,
=======
                   patience = 20,
                   verbose = 1)

hist = model.fit(x_train, y_train, epochs = 3000,
                 batch_size = 1 ,
          validation_split = 0.18, verbose = 2,
>>>>>>> faa0e1cfdd5fe7c1df8683dad05a659a5e99e189
          callbacks=[es])

#4
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

submission_csv['count'] = y_submit
submission_csv.to_csv(path+"submission_0110_es_res.csv",
                      index = False)

print('loss:', loss)
print('r2:', r2)
'''
plt.figure(figsize=(9,6))   # figsize=(가로, 세로) inch단위
# plt.plot(hist.history['loss'], color = 'red',
        #  label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue',
         label = 'val_loss', marker = ',')
plt.legend(loc='upper right')
plt.title('따릉이 로스값')
plt.xlabel('epoch')
plt.ylabel('val_loss')
plt.grid()
plt.show()
'''

<<<<<<< HEAD
## val_loss best ##
# val_loss: val_loss: 1888.3268

## R2 best ##
# R2: 0.5695930625541372

# loss: 3318.330810546875
# r2: 0.5183440912593621



# restore   69.7278194041
# loss: 2624.61181640625
# r2: 0.558611087447247
=======

## val_loss best ##
# val_loss: val_loss: 1888.3268

## R2 best ##
# R2: 0.5695930625541372
>>>>>>> faa0e1cfdd5fe7c1df8683dad05a659a5e99e189
