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

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

#1

np_path = 'c:/_data/_save_npy/'

x = np.load(np_path +'keras39_5_x_train.npy')
y = np.load(np_path + 'keras39_5_y_train.npy')
test = np.load(np_path + 'keras39_5_x_test.npy')
# y_test = np.load(np_path + 'keras39_5_y_test.npy')

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15, stratify=y, random_state=0)

img_path = 'c:/_data/kaggle/men_women/me.JPG'
img = Image.open(img_path)
img = img.resize((200,200))
img_array = np.array(img)

img_array = preprocess_input(img_array.reshape(1,200,200,3))

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

y_predict = model.predict(img_array)

print('loss', results[0])
print('acc', results[1])
# print(y_predict)

print("Predicted class probability:", y_predict[0, 0])  # 0이 남자

# loss 0.6826872825622559
# acc 0.573440670967102
# Predicted class probability: 0.9821331

# loss 0.6882096529006958
# acc 0.5714285969734192
# Predicted class probability: 0.0

# loss 0.6825845241546631
# acc 0.573440670967102
# Predicted class probability: 0.56206274

# loss 0.6824924349784851
# acc 0.573440670967102
# Predicted class probability: 1.0

# loss 0.6821460127830505
# acc 0.5754527449607849
# Predicted class probability: 2.2151606e-25

# loss 0.6825656890869141
# acc 0.573440670967102
# Predicted class probability: 0.00032575772