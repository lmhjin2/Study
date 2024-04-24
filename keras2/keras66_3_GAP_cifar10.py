import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.optimizers import *
import time as tm

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# acc = 0.77 이상
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train/255. # 0~255까지 있는 데이터라 255로 나눠서 mimaxscaler랑 같은효과
x_test = x_test/255. 

#2
model = Sequential()

model.add(Conv2D(256, kernel_size=(2,2), input_shape=(32,32,3),
                 padding='same', strides=2,
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) # 사실상 기본값임. pool_size(2,2)가 기본. strides는 pool_size와 같은게 기본.
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=(2,2), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size=(2,2), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(10, activation='softmax'))

#3
es = EarlyStopping(monitor='val_accuracy', mode = 'auto',
                   patience = 50, verbose = 1,
                   restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer = Adam(learning_rate=0.01),
              metrics=['accuracy'])
start_time = tm.time()
model.fit(x_train, y_train, epochs = 1000, batch_size = 1000, 
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

# run time 294.33
# loss 1.1299116611480713
# acc  0.6051999926567078 0.6052  

# run time 346.52
# loss 1.320745825767517
# acc  0.5952000021934509 0.5952

# GAP
# run time 454.8
# loss 0.8186299204826355
# acc  0.7142000198364258 0.7142
