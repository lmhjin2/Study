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

#1
np_path = 'c://_data//_save_npy//'

x_train = np.load(np_path + 'keras39_3_x_train.npy')
y_train = np.load(np_path + 'keras39_3_y_train.npy')
x_test = np.load(np_path + 'keras39_3_x_test.npy')
y_test = np.load(np_path + 'keras39_3_y_test.npy')


#2
model = Sequential()
model.add(Conv2D(2, (16, 16), input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(2, (12, 12), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(1, (8, 8), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


#3
model.compile(loss = 'binary_crossentropy', optimizer='adam',
              metrics = ['accuracy'])
es = EarlyStopping(monitor='val_loss', mode = 'auto',
                   patience = 50, verbose = 1,
                   restore_best_weights=True)
fit_start = tm.time()
hist = model.fit(x_train, y_train, epochs= 1000, batch_size = 64,
          verbose= 1, validation_split=0.2, callbacks=[es])
fit_end = tm.time()
fit_time = np.round(fit_end - fit_start, 2)

#4
loss = model.evaluate(x_test, y_test, verbose=1)

print('fit time', fit_time)
print('loss', loss)

# fit time 87.31
# loss [0.6893544793128967, 1.0]