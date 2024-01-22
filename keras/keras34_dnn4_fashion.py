import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import time as tm

# 0.99 이상 -> 0.95 이상
#1
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)
print(np.unique(y_train, return_counts=True))
# 0부터 9까지 각각 6000개씩

# import matplotlib.pyplot as plt
# plt.imshow(x_train[1], "RdGy_r")
# plt.show()
# x_train = x_train.reshape(-1,28,28,1)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000,28,28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
# print(x_train.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = x_train.reshape(60000, 784)

#2
model=Sequential()
model.add(Dense(800, input_shape=(784,), activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(700, activation='relu'))
model.add(Dense(570, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(420, activation='relu'))
model.add(Dense(340, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3
es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience = 50, verbose = 1, restore_best_weights = True)
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
start_time=tm.time()
model.fit(x_train, y_train, epochs = 2000, batch_size = 500,
          verbose=1, validation_split = 0.21 , callbacks=[es])
end_time=tm.time()
run_time = round(end_time-start_time, 2)

#4
results = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

y_train = np.argmax(y_train, axis = 1)
y_test = np.argmax(y_test, axis = 1)
y_predict = np.argmax(y_predict, axis = 1)

acc = accuracy_score(y_test, y_predict)

print('run time ', run_time)
print('loss', results[0])
print('acc', results[1], acc)


# run time  159.16
# loss 0.23071454465389252
# acc 0.9204999804496765 0.9205

# run time  63.41
# loss 0.22961604595184326
# acc 0.9218000173568726 0.9218

# 위가 기본
# ==================================================================
# 아래가 padding + strides

# run time  46.13
# loss 0.3341246247291565
# acc 0.8773000240325928 0.8773

# run time  53.36
# loss 0.23424088954925537
# acc 0.916100025177002 0.9161

# ====================================================================
# 아래는 no MaxPooling

# run time  49.1
# loss 0.2532179653644562
# acc 0.9108999967575073 0.9109



# run time  56.22
# loss 0.45052406191825867
# acc 0.8392000198364258 0.8392

# run time  66.9
# loss 0.40729230642318726
# acc 0.8503000140190125 0.8503