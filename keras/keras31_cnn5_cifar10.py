import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import time as tm

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)
# print(np.unique(y_train, return_counts=True))
    # (array([0,     1,     2,    3,    4,    5,    6,   7,     8,    9], dtype=uint8),
    #  array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
    #   dtype=int64))

# acc = 0.77 이상
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

import matplotlib.pyplot as plt
plt.imshow(x_train[5], "gray")
plt.show()

x_train = x_train/255 # 0~255까지 있는 데이터라 255로 나눠서 mimaxscaler랑 같은효과
x_test = x_test/255 



#2
model = Sequential()

model.add(Conv2D(50, kernel_size=(2,2), input_shape=(32,32,3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(50, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(50, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(220))
model.add(Dropout(0.2))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# model.add(Conv2D(8, kernel_size=(3,3), input_shape=(32,32,3), activation='sigmoid'))
# model.add(Conv2D(4, kernel_size=(2,2), activation='relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(3,3)))
# model.add(Conv2D(6, kernel_size=(3,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='softmax'))
# model.summary()

#3
es = EarlyStopping(monitor='val_accuracy', mode = 'auto',
                   patience = 5000, verbose = 1,
                   restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer = 'adam',
              metrics=['accuracy'])
start_time = tm.time()
model.fit(x_train, y_train, epochs = 100000, batch_size = 1818, 
          verbose = 1 , validation_split = 0.18 , callbacks=[es])
end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)

y_train = np.argmax(y_train, axis = 1)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis = 1)

acc = accuracy_score(y_test, y_predict)

print('run time', run_time)
print('loss', results[0])
print('acc ', results[1], acc)



# run time 6824.01
# loss 0.6461122035980225
# acc  0.7796000242233276 0.7796
