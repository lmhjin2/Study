import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

text = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

# print(token.word_index)
    # {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
    # 많이 나올수록 앞순서, 같은 갯수면 앞에서 부터. 딕셔너리
# print(token.word_counts)
    # OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])
    # 앞에서 부터 그 단어가 몇개나 쓰였는지 알려줌

x = token.texts_to_sequences([text])
# print(x)    # [[4, 2, 2, 3, 3, 5, 6, 7, 1, 1, 1, 8]] 리스트

x1 = to_categorical(x)
# print(x1)
# print(x1.shape) # 1, 12, 9

##1 to_categorical에서 첫번째 0 을빼
x1 = x1[:,:,1:]
# print(x1)
# print(x1.shape) # (1, 12, 8)

##2 사이킷런 원핫인코더

x2 = np.array(x).reshape(-1,1)
# x2 = x2.T

ohe = OneHotEncoder(sparse=False)
x_ohe = ohe.fit_transform(x2)
# print(x_ohe)
# print(x_ohe.shape)

# [[0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]]
# (12, 8)


#3 판다스 겟더미
x3 = np.array(x).reshape(-1)
pd_x = pd.get_dummies(x3)
    # pd_x = pd_x.astype(int)   # (12,8)
print(pd_x)
print(pd_x.shape)   # (12, 8)


