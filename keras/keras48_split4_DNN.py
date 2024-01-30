# DNN 으로 구성

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

a = np.array(range(1,101))   # a = (1~100)
x_predict = np.array(range(96, 106)) # (96~105) 10개

size = 5    # x데이터는 4개 y데이터는 1개

def split_x(dataset, size):
    aaa = []            # aaa list만들기
    for i in range(len(dataset) - size + 1):        # dataset - 5 + 1
        subset = dataset[i : (i+size)]  # dataset에서 i 부터 i+5, 실질적으로는 i+4까지 subset 이라고 하기
        aaa.append(subset)          # subset에 담긴걸 aaa에 담기
      # aaa.append(dataset[i : (i+size)])  / # dataset에서 i 부터 i+5, 실질적으로는 i+4까지 aaa에 담기
    return np.array(aaa)   # 외주 맡긴거 돌려받기. 안주면 내가 못쓰자너. 쿠팡맨.

bbb = split_x(a, size)
# print(bbb)
# print(bbb.shape)  # (6, 5)

x = bbb[ : , : -1 ]  # "bbb" 배열에서 마지막 열을 제외한 모든 열을 가져와 "x"라는 이름의 새 배열에 저장.
    # 첫번째 : 은 행. 비어있으니 모든행 선택. / 두번째 : 은 열. : -1 이니까 마지막 하나 빼고 다.
y = bbb[ : , -1 ]   # "bbb" 배열에서 마지막 열만 가져와 "y"라는 이름의 새 배열에 저장.
    # 첫번째 :은 행. 모든행/ 두번째 -1 은 인덱스 번호 [-1]. 마지막 하나만 갖고오기

# 모델 구성 및 평가 예측

xxx = split_x(x_predict, size)
predict = xxx[:,:-1]
predict = np.append(predict, [[102,103,104,105]], axis=0)
print(predict.shape)    # (7, 4)
print(x.shape, y.shape) # (96,4) (96,)
# x = x.reshape(-1,4,1)
# predict = predict.reshape(-1,4,1)

#2
model = Sequential()
model.add(Dense(32, input_shape =(4,), activation='swish'))
model.add(Dense(15, activation='swish'))
model.add(Dense(48, activation='swish'))
model.add(Dense(26, activation='swish'))
model.add(Dense(12, activation='swish'))
model.add(Dense(1))

model.summary()

#3
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='auto',
                   patience = 200, verbose=1, restore_best_weights=True)
model.fit(x, y, epochs = 1000, batch_size=32, callbacks=[es])

#4
loss = model.evaluate(x, y)
y_predict = model.predict(predict)
print('loss', loss)
print(np.round(y_predict),"\n", y_predict)

# [[100.]
#  [101.]
#  [102.]
#  [103.]
#  [104.]
#  [105.]
#  [106.]]
#  [[100.01467 ]
#  [101.01712 ]
#  [102.01974 ]
#  [103.02245 ]
#  [104.025314]
#  [105.028305]
#  [106.03141 ]]