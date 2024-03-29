import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler,StandardScaler
from PIL import ImageFile
import time as tm
import os
#1
np_path = 'c:/_data/_save_npy/'
path = 'c:/_data/kaggle/cat_and_dog/'
img_path = 'c:/_data/image/cat_and_dog/'
x = np.load(np_path + 'keras39_3_x_train.npy')
y = np.load(np_path + 'keras39_3_y_train.npy')
test = np.load(np_path + 'keras39_3_test.npy')

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=0)

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

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# (25997, 100, 100, 3) (4000, 100, 100, 3)
# (25997,) (4000,)

# (45997, 100, 100, 3) (4000, 100, 100, 3)
# (45997,) (4000,)

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
                   patience = 5, verbose = 1,
                   restore_best_weights=True)
fit_start = tm.time()
hist = model.fit(x_train, y_train, epochs= 300, batch_size = 64,
          verbose= 1, validation_split=0.2, callbacks=[es])
fit_end = tm.time()
fit_time = np.round(fit_end - fit_start, 2)

#4

# test = np.load(path_test) # ImageDataGenerator로 읽어오기
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(test)

print(y_predict.shape)
y_predict = np.around(y_predict.reshape(-1))
print(y_predict)

folder_dir = img_path + 'Test/test'
id_list = os.listdir(folder_dir)

for i, id in enumerate(id_list):
    id_list[i] = int(id.split('.')[0])

for id in id_list:
    print(id)


y_submit = pd.DataFrame({'id':id_list, 'Target':y_predict})
# print(y_submit)
y_submit.to_csv(path+"submit_0126.csv", index=False)

# submit_df = pd.DataFrame(submit, columns=['Class'])
# submit_df.to_csv('c:/_data/kaggle/cat_and_dog/')

print('fit time', fit_time)
print('loss', loss)

# fit time 87.31
# loss [0.6893544793128967, 1.0]

# fit time 37.68
# loss [0.6846022009849548, 0.5565000176429749]