import tensorflow as tf     # tensorflow 를 땡겨오고, tf 라고 줄여서 쓴다. or tf 라고 부르겠다
print(tf.__version__)    # 2.15.0       # warning 은 경고는 하는데 실행은 됨 / error는 실행도 안됨
from keras.models import Sequential  # Sequnetial 은 class 이름
from keras.layers import Dense   # Dense 레이어 (아직 설명 x) 를 땡겨옴
import numpy as np      # numpy를 땡겨오고 np라고 부르겠다  # 작업하는 수치들을 numpy 에 넣는다고 생각하면 편함

#1. 데이터                  
x = np.array([1,2,3])     # python의 list형태라 tensorflow에서 못씀. 그래서 추가된게 5번줄 코드.
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()    # 모델이라는 이름의 모델을 정의할거다 / 순차적 모델            x 한덩어리 y 한덩어리 불러오는거임
model.add(Dense(1, input_dim=1))  # 덴스형태 y = wx + b  /  dim = dimension 차원  /  x 데이터 한덩어리를 하나의 차원으로 인식함  /  처음 1이 y 값 마지막 1 이 x 값

#3. 컴파일, 훈련                                            # mse 의 s 는 squre=제곱. 제곱하는 방식으로 거리를 재겠다는 뜻
model.compile(loss='mse', optimizer='adam') # 두개의 값(=loss) 의 차이는 mse를 쓰겠다는 말 / mse는 어떤 수식임 # loss를 건드려주는게 옵티마이저.adam만 써도 85퍼는 됨
model.fit(x, y, epochs=2200)  # 최적의 weight가 생성  # x가 input_dim = 1 에 연결, y가 앞에 1에 연결 -> (1, input_dim=1) 에서 앞에 1이 9번줄 y로 뒤에 1이 8번줄 x로 연결

#4. 평가, 예측
loss = model.evaluate(x,y)
print("로스 : ", loss)
result = model.predict([4])
print("4의 예측값 : ", result)

# 최초에 모델이 랜덤값을 던지고 수를 맞춰감. 나중에 랜덤값 지정하는법 배움