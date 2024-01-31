import numpy as np
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, concatenate, Concatenate
from sklearn.model_selection import train_test_split


#1 data
x1_datasets = np.array([range(100), range(301,401)]).T    # 삼성 종가, 하이닉스 종가 [라고생각하기]

y1 = np.array(range(3001, 3101)) # 비트코인 종가 [라고생각하기]
y2 = np.array(range(13001, 13101)) # 이더리움 종가 [라고생각하기]

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, y1, y2, train_size = 0.7, random_state = 0 )

    # print(x1_train.shape, x2_train.shape, y_train.shape)    # (70, 2) (70, 3) (70,)
    # (name = ) 부분은 써도되고 안써도됨. 그냥 이름붙이는거임

# 2-1 model
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(10, activation='relu', name='bit2')(dense1)
dense3 = Dense(10, activation='relu', name='bit3')(dense2)
output1 = Dense(1, activation='relu', name='output1')(dense3)
output2 = Dense(1, activation='relu', name='output2')(dense3)

model1 = Model(inputs = input1, outputs = [output1, output2])
    
#3
model1.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', mode='min', patience = 300, verbose = 1, restore_best_weights=True)

model1.fit(x1_train, [y1_train,y2_train], epochs = 5000, batch_size = 32, verbose = 1, callbacks=[es])


#4
loss1 = model1.evaluate(x1_test, [y1_test, y2_test])


print('y1 loss : ', loss1)


# y1 loss :  [179606928.0, 9303465.0, 170303456.0]