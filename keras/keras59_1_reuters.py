from keras.datasets import reuters
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, GRU, Dropout, Flatten, Embedding, Reshape
from keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 1000,
                                                         test_split=0.2)
    # print(x_train)
    # print(x_train.shape, x_test.shape) # (8982,) (2246,)
    # print(y_train.shape, y_test.shape) # (8982,) (2246,)

    # print(type(x_train))        # <class 'numpy.ndarray'>
    # print(type(x_train[0]))     # <class 'list'>
    # print(len(x_train[0]), len(x_train[1]))     # 둘이 다름

    # print("뉴스기사의 최대 길이 :", max(len(i) for i in x_train))  # i에 x_train[n] 리스트가 들어가고 그 길이(len)을 측정. 다 돌고 최대값만 뽑기
    # 뉴스기사의 최대 길이 : 2376
    # print("뉴스기사의 평균 길이 :", sum(map(len, x_train)) / len(x_train))  # 알아서 찾아보란다.
    # 뉴스기사의 평균 길이 : 145.5398574927633

    # print(len(x_train[3]))
    # print(y_train)
    # print(np.unique(y_train)) # 0 ~ 45
    # print(len(np.unique(y_train))) # 46

# word_index = reuters.get_word_index()
# print(len(word_index))
# 30979

# 전처리
from keras.utils import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

# y 원핫은 선택사항. 귀찮으면 sparse_categorical_crossentropy
    # print(x_train.shape, x_test.shape)  # (8982, 100) (2246, 100)

#2
model = Sequential()
model.add(Embedding(input_dim = 100, output_dim=28))
model.add(LSTM(28, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.summary()

#3
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='loss', mode='auto', verbose=1,
                   patience= 500, restore_best_weights=True)
model.fit(x_train, y_train, batch_size = 1000, epochs = 3000, verbose=1, validation_data=(x_test,y_test), validation_split=0.2, callbacks=[es])

#4
loss, acc = model.evaluate(x_test, y_test)

print('loss', loss)
print('acc', acc)

# loss 1.7657196521759033
# acc 0.5538735389709473

