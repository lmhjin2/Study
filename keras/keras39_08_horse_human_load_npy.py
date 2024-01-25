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

# {'horses': 0, 'humans': 1}
#1
np_path = 'c:/_data/_save_npy/'
#1-1
# x_train = np.load(np_path + 'keras39_07_x_train.npy')
# y_train = np.load(np_path + 'keras39_07_y_train.npy')
# x_test = np.load(np_path + 'keras39_07_x_test.npy')
# y_test =np.load(np_path + 'keras39_07_y_test.npy')

#1-2
x = np.load(np_path + 'keras39_07_x.npy')
y = np.load(np_path + 'keras39_07_y.npy')

# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
            test_size=0.2, random_state=42,stratify=y)

# print(x_train.shape)    # (821, 300, 300, 3)
#2-1

# model = Sequential()
# model.add(Conv2D(4, (5,5), input_shape=(300,300,3),
#                  activation='relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())
# model.add(Conv2D(7,(6,6), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(3, (2,2), strides=(2,2), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(3, activation='relu'))
# model.add(Dense(7, activation='relu'))
# model.add(Dense(2, activation='softmax'))

# model.summary()

#2-2
model = Sequential()
model.add(Conv2D(32, kernel_size=(2,2), input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]) , activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.2))    

model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.3))
    
model.add(Conv2D(126, (2,2), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.summary()

#3
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics = ['accuracy'])
es = EarlyStopping(monitor='val_loss', mode = 'min',
                   patience = 100, verbose = 1,
                   restore_best_weights = True )
fit_start = tm.time()
hist = model.fit(x_train, y_train, epochs = 300, batch_size = 32,
                validation_split=0.2, callbacks = [es])
fit_end = tm.time()
fit_time = np.round(fit_end - fit_start, 2)

#4
loss,acc = model.evaluate(x_test,y_test)

print('loss', loss)
print('accuracy', acc)

# loss: 0.0466 - accuracy: 0.9903
# accuracy 0.9902912378311157

# loss: 0.0132 - cpu2accuracy: 1.0000