import numpy as np
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, concatenate, Concatenate
from sklearn.model_selection import train_test_split

#1 data
x1_datasets = np.array([range(100), range(301,401)]).T    # 삼성 종가, 하이닉스 종가 [라고생각하기]
x2_datasets = np.array([range(101,201),
                        range(411,511), range(150, 250)]).transpose()   # 원유, 환율, 금시세 [라고생각하기]
x3_datasets = np.array([range(100), range(301, 401),
                        range(77,177), range(33,133)]).transpose()      # 이젠 생각해 내기도 기찮음

    # print(x1_datasets.shape, x2_datasets.shape, x3_datasets.shape) # (100, 2) (100, 3) (100, 4)
y = np.array(range(3001, 3101)) # 비트코인 종가 [라고생각하기]

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y, train_size = 0.7, random_state = 0 )

    # print(x1_train.shape, x2_train.shape, y_train.shape)    # (70, 2) (70, 3) (70,)
    # (name = ) 부분은 써도되고 안써도됨. 그냥 이름붙이는거임

# 2-1 model1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='bit1')(input1)
dense2 = Dense(10, activation='relu', name='bit2')(dense1)
dense3 = Dense(10, activation='relu', name='bit3')(dense2)
output1 = Dense(10, activation='relu', name='bit4')(dense3)

# model1 = Model(inputs = input1, outputs = output1)
# model1.summary()    # total params 360

# 2-2 model2
input11 = Input(shape=(3,))
dense11 = Dense(100, activation='relu', name='bit11')(input11)
dense12 = Dense(100, activation='relu', name='bit12')(dense11)
dense13 = Dense(100, activation='relu', name='bit13')(dense12)
output11 = Dense(5, activation='relu', name='bit14')(dense13)

# model2 = Model(inputs = input11, outputs = output11)
# model2.summary()    # total params 21105

    # 모델 두개 output갯수가 다른데 어케 합칠래?
    # lastoutput = Dense(1,)([(output1)(output11)])
    # concatenate 와 Concatenate

# 2-3 model3
input21 = Input(shape=(4,))
dense21 = Dense(50, activation='relu', name = 'bit21')(input21)
dense22 = Dense(50, activation='relu', name = 'bit22')(dense21)
dense23 = Dense(50, activation='relu', name = 'bit23')(dense22)
output21 = Dense(15, activation='relu', name = 'bit24')(dense23)

# 2-4 concatenate
merge1 = concatenate([output1, output11, output21], name = 'mg1')     # 이건 소문자
# merge1 = Concatenate(name='mg1')([output1, output11])      # 이건 대문자
    #  mg1 (Concatenate)              (None, 15)           0           ['bit4[0][0]', 'bit14[0][0]']
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(11, name='mg3')(merge2)
last_output = Dense(1, name = 'last')(merge3)

model = Model(inputs=[input1, input11, input21], outputs = last_output)
# model.summary()
    # print(merge1.shape) # (None, 15)
#3
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience = 300, verbose = 1, restore_best_weights=True)
model.fit([x1_train, x2_train, x3_train], y_train, epochs = 5000, batch_size = 1, verbose = 1, callbacks=[es])

#4
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)

print('loss : ', loss)
# loss :  0.019153716042637825