# https://www.kaggle.com/playlist/men-women-classification

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


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#1

np_path = 'c:/_data/_save_npy/'

x_train = np.load(np_path +'keras39_5_x_train.npy')
y_train = np.load(np_path + 'keras39_5_y_train.npy')
x_test = np.load(np_path + 'keras39_5_x_test.npy')
y_test = np.load(np_path + 'keras39_5_y_test.npy')

# print(x_train.shape, y_train.shape) # (3309, 200, 200, 3) (3309,)
# print(x_test.shape, y_test.shape)   # (3309, 200, 200, 3) (3309,)

#2
model = Sequential()
model.add(Conv2D(4, (40,40), input_shape = (200,200,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(2, (30,30), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(2, (5,5), padding='valid', activation='relu'))
model.add(Conv2D(2, (9,9), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(42))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3
model.compile(loss = 'binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
es = EarlyStopping(monitor = 'val_loss', mode='auto',
                   patience = 50, verbose = 1,
                   restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs = 300, batch_size = 16,
                 validation_split = 0.2, callbacks=[es])

#4
results = model.evaluate(x_test, y_test)

# y_predict = model.predict(me)


print('loss', results[0])
print('acc', results[1])
# print(y_predict)
