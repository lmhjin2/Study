import numpy as np
from keras.datasets import mnist

#1 데이터
(x_train, _) , (x_test , _ ) = mnist.load_data()    # 비지도 학습을 할 것 이라서 y는 받아 오지만 _ 를 써서 비워둠

x_train = x_train.reshape(60000,28*28).astype('float32')/255.

x_test = x_test.reshape(10000,28*28).astype('float32')/255.

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

autoencoder.fit(x_train , x_train,       # 지금은 같은놈을 같은걸로 하는데 원래는 train으로 노이즈 먹은 train을 훈련하게 된다
                epochs = 10 , batch_size= 256 , validation_split=0.2)

#4 평가, 예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize = (20,4) )
for i in range(n) :
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

