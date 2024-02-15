# m31_1에서 뽑은4가지결과로 4가지모델을 만들어
# 154
# 331
# 486
# 713
# 784 <- 기본
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time as tm

(x_train, y_train), (x_test, y_test) = mnist.load_data()   # y가 필요가없어서 _넣음. 안넣으면 x에 다들어감
# x = np.append(x_train, x_test, axis=0)      # concate, append,vhstack 뭘쓰던 ㄱㅊ
# x = np.vstack([x_train, x_test])           # concate, append,vhstack 뭘쓰던 ㄱㅊ
x = np.concatenate([x_train, x_test], axis=0)      # concatenate, append,vhstack 뭘쓰던 ㄱㅊ
# print(x.shape)      # (70000, 28, 28)
y = np.concatenate([y_train, y_test], axis=0)      # concatenate, append,vhstack 뭘쓰던 ㄱㅊ
# print(y.shape)      # (70000,)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
xx = x
n_comp = [154, 331, 486, 713, 784]
########################### ver.1
# model = RandomForestClassifier()
# for i in n_comp:
#     pca = PCA(n_components=i)
#     xx = pca.fit_transform(xx)
#     x_train,x_test, y_train,y_test = train_test_split(xx,y,test_size=0.2, random_state=0,stratify=y, shuffle=True)
#     #3
#     start_time = tm.time()
#     model.fit(x_train,y_train)
#     end_time = tm.time()
#     #4
#     acc = model.score(x_test, y_test)
#     print(f"====================")
#     print(f"PCA={i}")
#     print(f"걸린시간={np.round(end_time-start_time,2)}")
#     print(f"ACC={acc}")
#     print(f"====================")
    
#     xx = pca.inverse_transform(xx)

# PCA=154
# 걸린시간=46.83
# ACC=0.9497857142857142
# PCA=331
# 걸린시간=52.56
# ACC=0.9464285714285714
# PCA=486
# 걸린시간=60.05
# ACC=0.9451428571428572
# PCA=713
# 걸린시간=67.77
# ACC=0.9405714285714286
# PCA=784
# 걸린시간=72.94
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# ACC=0.9409285714285714

########################### ver.2
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'auto', patience=50, verbose=1, restore_best_weights=True)
n_comp = [154, 331, 486, 713, 784]
for i in n_comp:
    pca = PCA(n_components=i)
    xx = pca.fit_transform(x)
    x_train = xx[:60000]
    x_test = xx[60000:]
    #2
    model = Sequential()
    model.add(Dense(128, input_shape=(i,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    #3
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    start_time = tm.time()
    model.fit(x_train,y_train, epochs=500, batch_size=3000, validation_split=0.2, verbose=0, callbacks=[es])
    end_time = tm.time()
    #4
    results = model.evaluate(x_test, y_test)
    print(f"====================")
    print(f"PCA={i}")
    print(f"걸린시간={np.round(end_time-start_time,2)}초")
    print(f"ACC={results[1]}")
    print(f"Loss={results[0]}")
    print(f"====================")
    
    xx = pca.inverse_transform(xx)


# PCA=154
# 걸린시간=7.02초
# ACC=0.921500027179718
# Loss=0.5660344958305359
# PCA=331
# 걸린시간=5.07초
# ACC=0.8946999907493591
# Loss=0.6228562593460083
# PCA=486
# 걸린시간=5.12초
# ACC=0.8912000060081482
# Loss=0.5913116335868835
# PCA=713
# 걸린시간=7.71초
# ACC=0.9003999829292297
# Loss=0.769172191619873
# PCA=784
# 걸린시간=6.79초
# ACC=0.9024999737739563
# Loss=0.5522394180297852


