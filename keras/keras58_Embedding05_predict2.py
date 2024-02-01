import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, GRU, Dropout, Flatten, Embedding, Reshape
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

######## [실습] #################

# 결과는 긍정? 부정?

#1 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밌네요.', 
    '상헌이 바보', '반장 잘생겼다', '욱이 또 잔다'
]
x_predict = '나는 정룡이가 정말 싫다. 재미없다 너무 정말'

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])  # 1 = 긍정 / 0 = 부정

all_text = docs.copy()
all_text.append(x_predict)
# print(all_text)

token = Tokenizer()
token.fit_on_texts(all_text)
# print(token.word_index)
# {'너무': 1, '참': 2, '재미없다': 3, '정말': 4, '재미있다': 5, '최고에요': 6, 
# '잘만든': 7, '영화에요': 8, '추천하고': 9, '싶은': 10, ' 영화입니다': 11, 
# '한': 12, '번': 13, '더': 14, '보고': 15, '싶어요': 16, '글쎄': 17, 
# '별로에요': 18, '생각보다': 19, '지루해요': 20, '연기가': 21, '어색해요': 22, 
# '재미없어요': 23, '재밌네요': 24, '상헌이': 25, '바보': 26, '반장': 27, 
# '잘생겼다': 28, '욱이': 29, '또': 30, '잔다': 31, '나는': 32, '정룡이가': 33, '싫다': 34}

word_size = len(token.word_index)

all_text_list = token.texts_to_sequences(all_text)
all_text_list = pad_sequences(all_text_list)

x = all_text_list[:len(docs)]
x_predict = all_text_list[len(docs):]


#2                  
model = Sequential()
model.add(Embedding(input_dim=word_size, output_dim = 10))
model.add(LSTM(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
#3
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
es = EarlyStopping(monitor='loss', mode='auto', verbose=1,
                   patience = 200, restore_best_weights=True)
model.fit(x, labels, batch_size = 32, epochs = 300, verbose=1, callbacks=[es])

#4
loss, acc = model.evaluate(x, labels)
y_predict = np.around(model.predict(x_predict)) # np.around
print('loss', loss)
print('acc', acc)
print('y_predict: ', y_predict)
# loss 5.836788659507874e-06
# acc 1.0





