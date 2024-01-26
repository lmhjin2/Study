import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img, to_categorical
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time as tm

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train/255.
x_test = x_test/255.

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # rotation_range=30,
    # zoom_range=0.2,
    # shear_range=10,
    fill_mode='nearest'
)

augment_size = 30000

randidx = np.random.randint(x_train.shape[0], size = augment_size)
        # np.random.randint(50000, 30000)   5만개중에 3만개의 숫자를 뽑아내라

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# 복사된 x 에 라벨을 넣으려고 y까지 하는거
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0]

print(x_train.shape, x_augmented.shape)    # (50000, 32, 32, 3) (30000, 32, 32, 3)
# 데이터 합치기. (merge)도 가능
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2
model = Sequential()

model.add(Conv2D(50, kernel_size=(2,2), input_shape=(32,32,3),
                 padding='same', strides=2,
                 activation='sigmoid'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) # 사실상 기본값임. pool_size(2,2)가 기본. strides는 pool_size와 같은게 기본.
model.add(Dropout(0.2))
model.add(Conv2D(50, kernel_size=(2,2), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(50, kernel_size=(2,2), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(220))
model.add(Dropout(0.2))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#3
es = EarlyStopping(monitor='val_accuracy', mode = 'auto',
                   patience = 10, verbose = 1,
                   restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer = 'adam',
              metrics=['accuracy'])
start_time = tm.time()
model.fit(x_train, y_train, epochs = 100, batch_size = 1818, 
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


# run time 137.75
# loss 1.1463711261749268
# acc  0.5996999740600586 0.5997







