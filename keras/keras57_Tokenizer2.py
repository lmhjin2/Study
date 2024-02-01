import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

text1 = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '상헌이가 선생을 괴롭힌다. 상헌이는 못생겼다. 상헌이는 마구 마구 못생겼다.'


token = Tokenizer()
token.fit_on_texts([text1, text2])

# print(token.word_index)
    # {'마구': 1, '진짜': 2, '매우': 3, '상헌이는': 4, '못생겼다': 5, '나는': 6, '맛있는': 7, 
    # '밥을': 8, '엄청': 9, '먹었다': 10, '상헌이가': 11, '선생을': 12, '괴롭힌다': 13}
    # 많이 나올수록 앞순서, 같은 갯수면 앞에서 부터. 딕셔너리 형태
# print(token.word_counts)
    # OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), 
    #              ('마구', 5), ('먹었다', 1), ('상헌이가', 1), ('선생을', 1), ('괴롭힌다', 1), 
    #              ('상헌이는', 2), ('못생겼다', 2)])
    # 앞에서 부터 그 단어가 몇개나 쓰였는지 알려줌
    
# texts = [text1 + text2]
x = token.texts_to_sequences([text1 + text2])
# x = token.texts_to_sequences(texts)
# print(x)    # [[6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 10, 11, 12, 13, 4, 5, 4, 1, 1, 5]] 리스트

x1 = to_categorical(x)
# print(x1)
# print(x1.shape) # (1, 21, 14)

##1 to_categorical에서 첫번째 0 을빼
x1 = x1[:,:,1:]
# print(x1)
# print(x1.shape) # (1, 21, 13)

##2 사이킷런 원핫인코더

# x2 = np.array(x).T
# x2 = x2.T
x2 = np.array(x).reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
x_ohe = ohe.fit_transform(x2)   # sparse=False 아니면 여기 뒤에 .toarray()
# print(x_ohe)
# print(x_ohe.shape)
# (21, 13)


#3 판다스 겟더미
x3 = np.array(x).reshape(-1)
pd_x = pd.get_dummies(x3)
    # pd_x = pd_x.astype(int)   # True/False 대신 0/ 1 로 뽑기 (20, 14)
# print(pd_x)
# print(pd_x.shape)   # (21, 13)


