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

x = token.texts_to_sequences(docs)

from keras.utils import pad_sequences
# from keras_preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, 
                      padding = 'pre' ,
                      maxlen = 5 ,
                      truncating = 'pre')

pad_x = pad_x.reshape(15,5,1)

#2
model = Sequential()
model.add(Conv1D(82, 2, input_shape = (5,1), activation='relu'))
model.add(Flatten())
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

# loss 0.0012286806013435125
# acc 1.0