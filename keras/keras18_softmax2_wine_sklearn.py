import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (178, 13) (178,)
# print(pd.value_counts(y))
    # 1    71
    # 0    59
    # 2    48
# print(np.unique(y, return_counts=True))
    # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()   # ()안에 sparse = False 를 쓰면 .toarray()를 쓰면 안됨 
# print(y)    # [0 0 0 0 0.... 1 1 1 1.... 2 2 2 2....]
y = y.reshape(-1, 1)
# print(y)    # [[0].... [1].... [2]]
y_ohe = ohe.fit(y)  
# print(y_ohe)    # OneHotEncoder() <-- ???
y_ohe = ohe.transform(y).toarray()  # sparse = False 를 쓰면 .toarray()를 쓰면 안됨
# print(y_ohe)    # [[1. 0. 0.]... [0. 1. 0.]....[0. 0. 1.]]
# print(y_ohe.shape)
# print(y_ohe.shape)  # (178, 3)
# 마지막 Dense(Layer)에 3 

x_train, x_test, y_train, y_test = train_test_split(
    x, y_ohe, stratify=y, train_size = 0.8, random_state = 0 )
# print(y_test)   # y_encoded 랑 독같이 생김
# print(y_test.shape) # (36,3) # y_encoded 랑 독같이 생김

# print(np.unique(y_test, return_counts=True))   
    # (array([0., 1.], dtype=float32), array([72, 36], dtype=int64))
# [0. 1. 0.] 이렇게 생겨서 0두개 1하나.   2:1 비율이 맞음

#2
model = Sequential()
model.add(Dense(120, input_dim = 13))
model.add(Dense(90))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))


#3
model.compile(loss = 'categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience = 200, verbose=1,
                   restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs = 1000,
                 batch_size = 96, validation_split = 0.2,
                 verbose = 1, callbacks=[es])

#4
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
# print(results) 치면 숫자 두개나오는데 보기 좋으라고 밑에처럼 하는거임.
print('loss:', results[0])
print('acc:', results[1])
# print(y_predict)    # [0.1 0.2 0.7] 의 형태
# print(y_test)       # [0 1 0] 의 형태
# print(y_predict.shape, y_test.shape)    # 둘다 (36, 3)

# OneHot은 axis 무조건 1
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

# print(y_test)
#   # [1 0 1 0 1 2 2 0 1 0 2 2 0 2 1 0 2 2 1 0 1 2 2 1 2 0 1 0 1 1 0 0 1 1 0 1]
# print(y_test.shape) # (36,)
# print(y_predict)
#   # [1 0 1 1 1 0 2 0 1 0 2 1 0 2 0 0 2 2 1 0 1 0 2 1 1 0 1 0 1 1 0 0 1 1 0 1]
# print(y_predict.shape)  # (36,)

acc = accuracy_score(y_predict, y_test)
print('accuracy_score :', acc)


# loss: 0.7279049158096313
# acc: 0.8611111044883728
# accuracy_score : 0.8611111111111112

