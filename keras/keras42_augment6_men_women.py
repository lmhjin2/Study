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

train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # rotation_range=30,
    # zoom_range=0.2,
    # shear_range=0.7,
    fill_mode='nearest'
)
augment_size = 2000
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

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (4812, 200, 200, 3) (497, 200, 200, 3)
# (4812,) (497,)

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
# {'men': 0, 'women': 1}

# loss 0.6827759742736816
# acc 0.573440670967102
# Predicted class probability: 0.6666423
