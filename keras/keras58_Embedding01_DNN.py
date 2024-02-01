from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, GRU, Dropout, Flatten
from keras.callbacks import EarlyStopping

#1 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밌네요.', 
    '상헌이 바보', '반장 잘생겼다', '욱이 또 잔다'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])  # 1 = 긍정 / 0 = 부정

token = Tokenizer()
token.fit_on_texts(docs)

# print(token.word_index)
    # {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, 
    # '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, 
    # '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, 
    # '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, 
    # '재미없다': 22, '재밌네요': 23, '상헌이': 24, '바보': 25, '반장': 26, 
    # '잘생겼다': 27, '욱이': 28, '또': 29, '잔다': 30}
x = token.texts_to_sequences(docs)
# print(x)
    # [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], 
    # [10, 11, 12, 13, 14], [15], [16], [17, 18], 
    # [19, 20], [21], [2, 22], [1, 23], 
    # [24, 25], [26, 27], [28, 29, 30]]
# print(type(x))  /   # <class 'list'>
# x = np.array(x)   # 차원이 달라서 에러뜨는거 보여주려고 쓴거

from keras.utils import pad_sequences
# from keras_preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, 
                    #   padding = 'pre' ,   # shape를 맞추기위해 임의의 숫자를 채우는것.
                      maxlen = 5 ,          # 최대 길이. 기본값이 최대길이를 알아서 잘 잡아줌.
                    #   truncating = 'pre'  # 잘린다면 어디를 자를것인가.
                      ) # padding, truncating 에 pre면 앞에 post면 뒤에 / 기본값 둘다 'pre'
# print(pad_x)
# print(pad_x.shape)  # (15, 5)


# dnn, lstm, conv1d

#2
model = Sequential()
model.add(Dense(82, input_shape = (5,), activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(77, activation='relu'))
model.add(Dense(31, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
es = EarlyStopping(monitor='loss', mode='auto', verbose=1,
                   patience = 200, restore_best_weights=True)
model.fit(pad_x, labels, batch_size = 32, epochs = 1000, verbose=1, callbacks=[es])

#4
loss, acc = model.evaluate(pad_x, labels)

print('loss', loss)
print('acc', acc)

# loss 1.651626746479451e-07
# acc 1.0





