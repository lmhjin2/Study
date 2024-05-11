import numpy as np
from keras.datasets import mnist
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)

#1 데이터
(x_train, _) , (x_test , _ ) = mnist.load_data()    # 비지도 학습을 할 것 이라서 y는 받아 오지만 _ 를 써서 비워둠

x_train = x_train.reshape(60000,28*28).astype('float32')/255.

x_test = x_test.reshape(10000,28*28).astype('float32')/255.

# train에 노이즈를 먹이고 train을 훈련하게 된다        # 평균 0 표준편차 0.1 인 정규분포
x_train_noised = x_train + np.random.normal(0,0.1, size = x_train.shape )
x_test_noised = x_test + np.random.normal(0,0.1, size = x_test.shape )

# 노이즈를 넣으면 1.1까지 생기게 되는데 이걸 방지하기 위해서 상한선을 정해준다 1까지
print(x_train_noised.shape,x_test_noised.shape)         # (60000, 784) (10000, 784)
print(np.max(x_train),np.min(x_train))                  # 1.0 0.0
print(np.max(x_train_noised),np.min(x_train_noised))    # 1.5104973747855874 -0.5803276345612517

# 범위를 한정한다는 함수 clip
x_train_noised = np.clip(x_train_noised , a_min=0 , a_max=1)
x_test_noised = np.clip(x_test_noised , a_min=0 , a_max=1)

print(np.max(x_train_noised),np.min(x_train_noised))    # 1.0 0.0
print(np.max(x_test_noised),np.min(x_test_noised))    # 1.0 0.0

#2 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img = Input(shape=(784,))
# 인코더 그냥 써서 성능이 좋은걸 쓰면 됨
# encoded = Dense(64,activation='relu')(input_img)
# encoded = Dense(32,activation='relu')(input_img)
# encoded = Dense(1,activation='relu')(input_img)   # 하면 좋을 수도 있는데 값이 소실이 많이 되어서 안좋을 확률이 매우 높음
encoded = Dense(1024,activation='relu')(input_img)

# 디코더
# decoded = Dense(784,activation='linear')(encoded)
decoded = Dense(784,activation='sigmoid')(encoded)
# decoded = Dense(784,activation='relu')(encoded)
# decoded = Dense(784,activation='tanh')(encoded)

# loss = mse // sigmoid 64랑 썻을때 제일 좋음 , 1024랑 sigmoid , relu 도 좋음
# loss = binary_crossentropy // 1024 , sigmoid

autoencoder = Model(input_img,decoded)

autoencoder.summary()

#3 컴파일,훈련
# autoencoder.compile(optimizer='adam' , loss='mse' )
autoencoder.compile(optimizer='adam' , loss='binary_crossentropy' )

autoencoder.fit(x_train_noised , x_train,
                epochs = 10 , batch_size= 256 , validation_split=0.2)

#4 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize = (20,4) )
for i in range(n) :
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test_noised[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
