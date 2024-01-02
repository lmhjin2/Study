# from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras
print("tf 버전 : ", tf.__version__)
print("keras 버전 : ", keras.__version__)  # ctrl + space = 자동완성 꺼내기

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])     # 일부러 4와 5 위치 바꿈

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x,y, epochs = 2000, batch_size = 6)    # x, y 에서 한개씩 잘라서 작업하겠다는 뜻 (1,2,3,4,5,6 각각 따로). 메모리 부족할때 잘라서 분할 작업 가능
                                                # 얼만큼 한번에 돌릴거냐 지정하는것. batch_size = 1 일때 이 코드에서는 600번 도는것.
                                                # 85퍼 이상의 경우 batch_size를 잘라쓰는게 좋음 / 기본값 32
#4. 평가, 예측
loss = model.evaluate(x,y)      # 인터프리터 언어, 위에서 부터 실행되는거라 전과 순서가 조금 바뀌어도 괜찮다.
results = model.predict([7])    
print("로스 : ", loss)
print("예측값 : ", results)

# batch_size 랑 dense 숫자 바꿔서 로스 0.32 이하로 뽑기

# dense 1-10000-1   batch_size = 6  epochs = 100
# 로스 :  0.32387882471084595
# 예측값 :  [[6.7811184]]

# dense 1-10*5-100*11-10*5-1    batch_size = 6  epochs = 100
# 로스 :  0.32381126284599304
# 예측값 :  [[6.802495]]

# dense 1-10*5-100*11-10*5-1    batch_size = 6  epochs = 1000
# 로스 :  0.32380834221839905
# 예측값 :  [[6.8]]

# dense 1-10*5-100*11-10*5-1    batch_size = 6  epochs = 2000
# 로스 :  0.3238089978694916
# 예측값 :  [[6.7999997]]









