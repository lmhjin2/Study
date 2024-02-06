from keras.models import Sequential
from keras.layers import Dense
import random as rn
import tensorflow as tf
import keras
import numpy as np

# print(tf.__version__)   # 2.9.0
# print(keras.__version__)    # 2.9.0
# print(np.__version__)   # 1.26.3

rn.seed(333)            # 이제 다 먹힘. tf랑 np 위에 둬야함.
tf.random.set_seed(123) # 텐서 2.9.0에서 먹힘. 2.15.0 에서 안먹힘
np.random.seed(321) 

#1 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2 모델
model = Sequential()
model.add(Dense(5,
                # kernel_initializer='zeros', 
                # bias_initializer='zeros',
                input_dim=1 ) )
model.add(Dense(5))
model.add(Dense(1))

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs = 100, verbose = 0 )

#4 평가 예측
loss = model.evaluate(x,y, verbose=0)
print('loss',loss)
results = model.predict([4], verbose=0)
print('4의 예측값:', results)
