import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(np.unique(x, return_counts=True))
#     # (array([-173., -166., -164., ..., 7168., 7172., 7173.]),
#     #  array([1, 2, 1, ..., 1, 1, 1], dtype=int64))
# print('=====================================================================')
# print(np.unique(y, return_counts=True))
    # (array([1, 2, 3, 4, 5, 6, 7]),
    #  array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
    #  dtype=int64))

# print(x.shape, y.shape) # (581012, 54)  (581012,)
# print(pd.value_counts(y))
    #   원래 순서       # 내가 걍 맞춘거
    # 2    283301       # 1    211840       
    # 1    211840       # 2    283301
    # 3     35754       # 3     35754
    # 7     20510       # 4      2747
    # 6     17367       # 5      9493
    # 5      9493       # 6     17367
    # 4      2747       # 7     20510

from keras.utils import to_categorical
y_ohe = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y_ohe, stratify=y, train_size = 0.8, random_state = 0 )

print(x_train)
print(y_train)

#2
model = Sequential()
model.add(Dense(120, input_dim = 54))
model.add(Dense(90))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(8, activation='softmax'))
# keras만 8개

#3
model.compile(loss = 'categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience = 100, verbose=1,
                   restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs = 1000,
                 batch_size = 50000, validation_split = 0.2,
                 verbose = 2, callbacks=[es])

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


# loss: 0.6590697169303894
# acc: 0.7154462337493896
# accuracy_score : 0.7154462449334354



