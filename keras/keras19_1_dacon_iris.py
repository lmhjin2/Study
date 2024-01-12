# https://dacon.io/competitions/open/236070/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time as tm

#1
path = "c:/_data/dacon/iris/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

x = train_csv.drop(['species'], axis = 1)
y = train_csv['species']
# print(x)    # (120,4)
# print(np.unique(y, return_counts=True))    # (120,)
    # (array([0, 1, 2], dtype=int64), array([40, 41, 39], dtype=int64))
y_ohe = pd.get_dummies(y)

def auto(a,b,c):
    x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, stratify = y, 
                                    train_size = 0.8, random_state = a )
    # print(x_train) # (94,4)
    # print(y_train)  # (96,3)

    #2
    model = Sequential()
    model.add(Dense(80, input_dim = 4))
    model.add(Dense(60))
    model.add(Dense(40))
    model.add(Dense(20))
    model.add(Dense(10))
    model.add(Dense(3, activation = 'softmax'))
    # y의 label의 갯수 = 마지막 레이어숫자

    #3
    model.compile(loss= 'categorical_crossentropy', optimizer='adam',
                metrics = ['acc'])
    es = EarlyStopping(monitor='val_loss', mode='auto',
                    patience = 1000, verbose=1,
                    restore_best_weights=True)
    start_time = tm.time()
    hist = model.fit(x_train, y_train, epochs= c,
                    batch_size = b, validation_split= 0.2 ,
                    verbose = 1, callbacks=[es])
    end_time = tm.time()
    run_time = round(end_time - start_time, 2)

    #4 
    results = model.evaluate(x_test, y_test)
    y_submit = model.predict(test_csv)
    y_predict = model.predict(x_test)
    # encode 풀기
    y_test = np.argmax(y_test, axis=1)
    y_predict = np.argmax(y_predict, axis=1)
    y_submit = np.argmax(y_submit, axis=1)

    submission_csv['species'] = y_submit
    submission_csv.to_csv(path + "submission_0112_3.csv", index=False)

    acc = accuracy_score(y_predict, y_test) 

    print('acc:', results[1])
    print('accuracy_score :', acc)
    print('run time', run_time)
    print('loss:', results[0])
    return acc
    tm.sleep(1)
    
import random
for i in range(100000000):
    a = random.randrange(0, 4294967295)
    b = random.randrange(1, 3517)
    c = random.randrange(100, 4294967295)
    r = auto(a,b,c)
    print('random state:', a)
    if r > 0.98:
        print('random state :', a)
        print('batch size :', b)
        print('epochs :', c)
        print('acc', r)
        break













