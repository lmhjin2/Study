# https://dacon.io/competitions/open/236068/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
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

x = x.values.reshape(-1, 2,2,2)
test_csv = test_csv.values.reshape(-1, 2,2,2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state = 1 )




#2
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(2,2,2), padding='same',
                 activation='sigmoid'))
model.add(Conv2D(15, (2,2), padding='same', activation='relu'))
model.add(Conv2D(7, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience = 1, verbose = 1,
                   restore_best_weights = True)
import time as tm
start_time = tm.time()
hist = model.fit(x_train, y_train, epochs = 1000,
                 batch_size = 25, validation_split = 0.13,
                 verbose = 3)

end_time = tm.time()
run_time = round(end_time - start_time, 2)
#4
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)

submission_scv['Outcome'] = np.round(y_submit)
submission_scv.to_csv(path + "submission_0118_1.csv", index = False)

def ACC(y_test, y_predict):
    return accuracy_score(y_test, y_predict)

acc = ACC(y_test, np.round(y_predict))
print('loss', loss)
print('acc:', acc)
print("run time:", run_time)

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

# CPU
# 32.68 초

# GPU
# 47.07 초

# CNN
# loss [0.9040436744689941, 0.7755101919174194]
# acc: 0.7755102040816326
# run time: 56.14

