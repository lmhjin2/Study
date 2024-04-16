# Flatten 대신 사용
# Flatten을 하면 너무 커진다 - 각 필터마다 avergepooling을 해서 노드를 줄임
# 이미지사이즈가 상관없어지고 필터수에 따라 정해진다 (이미지내에 n빵을 통해 한값만 나오기 때문)

# 필터안에 값을 전부 n빵으로 한값을 뽑는다 
# 최근 Flatten 보다 성능이 좋은결과가 많이 나온다 


import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint


#1.data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape,y_train.shape)  #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    #(10000, 28, 28) (10000,)
# print(x_train)
# print(x_train[0])
print(y_train[0])   #5
print(np.unique(y_train,return_counts=True))    #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

#3차원 4차원으로 변경
x_train=x_train.reshape(60000,28,28,1)  #data 내용,순서 안바뀌면 reshape 가능

# x_test=x_test.reshape(10000,28,28,1)  #아래와 같다 - 값을 모를때 적용
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print(x_train.shape[0]) #60000
print(x_train.shape,x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

#2.model
model = Sequential()
model.add(Conv2D(100, (2,2),
                 strides=1,
                 padding='same',    #전 사이즈를 그대로 유지
                #  padding='valid',   #디폴트
                 input_shape=(10,10,1)))
model.add(MaxPooling2D())   #n빵
#(N, 5, 5, 10)로 바뀜
model.add(Conv2D(100,(2,2)))    # (N, 4, 4, 100)
model.add(Conv2D(100,(2,2)))    # (N, 3, 3, 100)
# model.add(Flatten())  
model.add(GlobalAveragePooling2D()) #(N, 100)
model.add(Dense(units=50))          #(N, 50) - 5050
# model.add(Dense(7,input_shape=(8,)))
# model.add(Dense(6))
model.add(Dense(10,activation='softmax'))    #숫자를 찾는 분류모델 / (N,27,27,10)
model.summary()

# Total params: 86,260 - GlobalavergePooling
# Total params: 126,260 - Flatten
# 연산량 차이가 큼


'''
#3.compile,fit
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=100,batch_size=32,verbose=1,validation_split=0.2)

#4.evaluate,predict
results=model.evaluate(x_test,y_test)
print("loss:",results[0])
print("acc:",results[1])
'''

# loss: 0.0266 - accuracy: 0.9930

#  run time 135.32
# loss 0.16735179722309113
# acc 0.979200005531311 0.9792

# run time 34.02
# loss 0.0694819912314415
# acc 0.9853000044822693 0.9853