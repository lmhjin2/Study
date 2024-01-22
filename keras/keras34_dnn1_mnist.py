import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)
# print(x_train)

x_train = x_train.reshape(60000, 28*28)
# print(x_train.shape[0]) # 60000
# x_test = x_test.reshape(10000, 28, 28, 1)
# print(x_test.shape[0]) # 10000

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
# print(x_train.shape, x_test.shape)
# (60000, 784) (10000, 784)

# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)


#2
model = Sequential()
model.add(Dense(800, input_shape=(784,), activation='relu'))
model.add(Dense(900, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(900, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(900, activation='relu'))
model.add(Dense(784, activation='softmax'))


#3
es = EarlyStopping(monitor = 'val_loss', mode = 'auto',
                   patience = 50, verbose=1, restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, validation_split = 0.2, 
          batch_size = 3000, verbose = 1, epochs = 500)


#4 
results = model.evaluate(x_test, y_test)
print('loss', results[0])
print('acc', results[1])

# loss 0.2325105518102646
# acc 0.9404000043869019

# loss 0.18292994797229767
# acc 0.9462000131607056

# loss 0.15299376845359802
# acc 0.9570000171661377


# Dense Layer
# loss 0.26949891448020935
# acc 0.9765999913215637





