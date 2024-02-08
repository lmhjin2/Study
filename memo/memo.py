import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Conv2D, SimpleRNN, LSTM, GRU, Dropout, Flatten, Embedding, Reshape, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import matplotlib as plt


## pandas -> numpy 변환
    # pandas = pandas.values
         

### 내용 수정, data 수정

## list
# .index()  괄호() 안에 이름으로 호출
# ex)
    # my_list = [1, 2, 3, 4, 5]
    # index = my_list.index(5)  # 5의 인덱스를 찾습니다.
    # my_list[index] = 10  # 해당 인덱스의 값을 변경합니다.
    # print(my_list)  # [1, 2, 3, 4, 10]
    # # 또는 index 번호로 호출
    # my_list[0] = 10  # 해당 인덱스의 값을 변경합니다.
    # print(my_list)  # [10, 2, 3, 4, 5]  / 윗줄 같이 쓰면 [10, 2, 3, 4, 10]


## tuple
# 튜플은 순서가있고 변경할 수 없는 자료형. [start:stop:step]형식으로 표현.
# [:] 안에 숫자는 인덱스 번호로 씀
# ex) index_num = (0, 1, 2, 3, 4)
    # my_tuple =  (1, 2, 3, 4, 5)
    # new_tuple = my_tuple[:2] + (10,) + my_tuple[3:]  # 새로운 튜플을 생성하여 요소를 대체합니다.
    # # 튜플은 변경이 불가능해서 새로 만들어야함. 
    # # my_tuple[:2] 에서 인덱스 번호 0,1을 불러오고 my_tuple[3:]에서 인덱스 번호 3 4 를 불러옴.
    # # 튜플은 괄호로 묶여야 하므로 (10,)  콤마는 단일 항목을 가진 튜플을 만들때 필요함
    # print(new_tuple)  # (1, 2, 10, 4, 5)

## dictionary
# 키 를 불러서 호출
# ex)
    # my_dict = {'a': 1, 'b': 2, 'c': 3}
    # my_dict['b'] = 10  # 'b' 키에 해당하는 값을 대체합니다.
    # print(my_dict)  # {'a': 1, 'b': 10, 'c': 3}

## .replace() str
# 문자열 객체의 내장 메서드(method). str 형태에만 사용가능
# ex)
    # sentence = "나는 사과를 좋아합니다."
    # new_sentence = sentence.replace("사과", "바나나")
    # print(new_sentence)

## 간단한 연산자
# 제곱 = **
# 나머지만 구할때 = %
# 몫만 구할때 = //


## 문자열 곱하기
    # print("="*50)
    # ==================================================

## 문자열 포매팅
# jump to python 63~76










