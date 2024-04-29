import numpy as np
import pandas as pd
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import *
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import time as tm

# accuracy 0.4 이상

# 1
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3), (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3), (10000, 1)
# print(np.unique(y_train, return_counts=True))
# 0 부터 99 전부 500개

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], "gray")
# plt.show()

# 2
model = Sequential()
model.add(Conv2D(512, (2,2), input_shape=(32,32,3), 
                 padding='same', strides=2, activation='relu'))
# (16, 16, 24) -> (32, 32, 24)
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(384, (2,2), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(256, (2,2), activation='relu'))
model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))  
model.add(GlobalAveragePooling2D())
model.add(BatchNormalization())
model.add(Dense(198, activation='relu'))  
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# model.summary()

#3
es = EarlyStopping(monitor = 'val_accuracy', mode='auto',
                   patience = 50, verbose = 1, restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate = 1e-3 ),
              metrics = ['accuracy'])

start_time = tm.time()

model.fit(x_train, y_train, epochs = 1000, batch_size = 500,
          verbose = 1, validation_split = 0.15, callbacks = [es])  

end_time = tm.time()
run_time = round(end_time-start_time, 2)

#4
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_train = np.argmax(y_train, axis = 1)
y_test = np.argmax(y_test, axis = 1)
y_predict = np.argmax(y_predict, axis = 1)

acc = accuracy_score(y_test, y_predict)

print('run time', run_time)
print('loss', results[0])
print('acc', results[1], acc)

# run time 217.21
# loss 2.3018598556518555
# acc 0.4108000099658966 0.4108

# run time 429.32
# loss 2.28546142578125
# acc 0.4097999930381775 0.4098

# run time 47.33
# loss 2.8813202381134033
# acc 0.29910001158714294 0.2991

# GAP
# run time 290.44
# loss 2.4405648708343506
# acc 0.445499986410141 0.4455