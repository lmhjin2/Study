import numpy as np
import pandas as pd
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import time as tm

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3), (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3), (10000, 1)
# print(np.unique(y_train, return_counts=True))
# 0 부터 99 전부 500개

x_train = x_train/255.
x_test = x_test/255.

train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest'
)
augment_size = 30000
randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle=False,
).next()[0]

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(y_test.shape, y_train.shape)  # (10000, 100) (80000, 1)
print(np.unique(y_train, return_counts=True))
# ohe = OneHotEncoder()
# ohe.fit(y_train)
# y_train = ohe.transform(y_train)
# y_test = ohe.transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_test.shape, y_train.shape)  # (10000, 100) (80000, 1)

# 2
model = Sequential()
model.add(Conv2D(24, (2,2), input_shape=(32,32,3), 
                 padding='same', strides=2, activation='sigmoid'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(45, (2,2), padding='same', activation='relu'))
model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(31, (2,2), activation='relu'))
model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(82, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

#3
es = EarlyStopping(monitor = 'val_accuracy', mode='auto',
                   patience = 200, verbose = 1, restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])

start_time = tm.time()

model.fit(x_train, y_train, epochs = 300, batch_size = 500,
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