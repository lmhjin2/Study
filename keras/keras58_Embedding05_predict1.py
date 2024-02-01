import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, GRU, Dropout, Flatten, Embedding, Reshape
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer

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
pad_x = pad_sequences(x,      # 사실상 x 만 넣고 나머지 기본값임.
                      padding = 'pre' ,
                      maxlen = 5 ,
                    truncating = 'pre')

# print(pad_x.shape)   # (15, 5)
# pad_x = pad_x.reshape(15, 5, 1)     # 2차원을 LSTM에 넣으면 알아서 돌더라. 주석처리 해도 잘됨


######################################################################################3333333333333333333333333333333333


#2                   # 어휘의 크기 ,  아웃풋은 맘대로 설정, 
model = Sequential() # 단어사전의 갯수,덴스레이어의 아웃풋, 인풋의 크기
# model.add(Embedding(input_dim=30, output_dim = 10, input_length = 5)) # output shape = (n, 5, 10)
# model.add(Embedding(31, 100)) # 잘돌아감
# model.add(Embedding(31,100,5)) # 에러
model.add(Embedding(input_dim=30, output_dim = 100, input_length=5)) # output shape = (n, 5, 10)
      # 엠베딩연산량 = input_dim * output_dim = 31 * 100 = 3100
      # 임베딩 인풋의 shape : 2차원, 임베딩 아웃풋 shape : 3차원
      # input_length 는 모르겠으면 그냥 안써도됨. 어차피 맞춰줌.
      # input_length 에는 약수만들어감. ex) 5 = 1, 5 / 6 = 1,2,3,6 / 7 = 1,7
      # input_dim = 31 # 기본값 , input_dim = 29, 단어사전의 갯수보다 작을때
      # input_dim = 31 # 기본값 , 
      # input_dim = 29, 단어사전의 갯수보다 작을때 : 연산량 줄어. 단어사전에서 임의로 빼. 성능 조금 저하
      # input_dim = 40 단어사전의 갯수보다 클때 : 연산량 늘어, 임의의 랜덤 임베딩 생성. 성능 조금 저하
model.add(LSTM(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
#3
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
es = EarlyStopping(monitor='loss', mode='auto', verbose=1,
                   patience = 200, restore_best_weights=True)
model.fit(pad_x, labels, batch_size = 32, epochs = 300, verbose=1, callbacks=[es])

#4
loss, acc = model.evaluate(pad_x, labels)

print('loss', loss)
print('acc', acc)

# loss 5.836788659507874e-06
# acc 1.0

######## [실습] #################
x_predict = '나는 정룡이가 정말 싫다. 재미없다 너무 정말'

# 결과는 긍정? 부정?

x_predict_seq = token.texts_to_sequences(x_predict)
pad_x_predict = pad_sequences(x_predict_seq, padding='pre', maxlen=5, truncating='pre')
result = model.predict(pad_x_predict)
print(result)

