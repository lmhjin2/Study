import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler,StandardScaler
from sklearn.metrics import accuracy_score
import time as tm
import os

# {'paper': 0, 'rock': 1, 'scissors': 2}

#1'
np_path = 'c:/_data/_save_npy/'
x = np.load(np_path + 'keras39_09_x.npy')
y = np.load(np_path + 'keras39_09_y.npy')

# print(x.shape, y.shape) # (2520, 150, 150, 3) (2520, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y )

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

augment_size = 1500
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

#2
model = Sequential()
model.add(Conv2D(4, (5,5), input_shape=(150,150,3),
                 activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(7,(6,6), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(3, (2,2), strides=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(3, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

#3
model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])
es = EarlyStopping(monitor = 'val_loss', mode='min',
                   patience=50, verbose=1,
                   restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs = 300, batch_size = 32,
                 validation_split=0.2, callbacks=[es])

#4
loss,acc = model.evaluate(x_test, y_test)

print('loss',loss)
print('accuracy',acc)

# loss 8.23384543764405e-05
# accuracy 1.0

