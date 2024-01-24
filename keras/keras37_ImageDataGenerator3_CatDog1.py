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

# Truncated File 어쩌구 warning 뜰때.

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

gen_start = tm.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range = 0.2,
    # shear_range=0.2,
    # fill_mode = 'nearest'
    )
test_datagen = ImageDataGenerator(
    rescale=1./255)

path_test = 'c://_data//image//cat_and_dog//Test//'
path_train = 'c://_data//image//cat_and_dog//Train//'

xy_train_data = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),
    batch_size = 200, 
    class_mode = 'binary',
    shuffle=True )

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size = 200, 
    class_mode = 'binary')


x = []
y = []
failed_i = []

for i in range(int(20000/200)):
    try:
        xy_data = xy_train_data.next()
        new_x = xy_data[0]
        new_y = xy_data[1]
        if i == 0:
            x = np.array(new_x)
            y = np.array(new_y)
            continue
        x = np.vstack([x, new_x])
        y = np.hstack([y,new_y])
        print("i :", i)
        print(f"{x.shape=}\n{y.shape=}")
    except:
        print("failed i:", i)
        failed_i.append(i)

print(failed_i)

print(f"{x.shape=}\n{y.shape=}")


load_start = tm.time()
print((xy_train_data.next()))
load_end = tm.time()
load_time = np.round(load_end-load_start, 2)
print('load time', load_time, '초')

gen_end = tm.time()
gen_time = np.round(gen_end - gen_start, 2)

                         # (batch, (target_size), 3)
# x_train = xy_train[0][0] # (batch, 150, 150, 3)
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]
r = int(np.random.uniform(1,1000))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                        random_state=r, stratify=y)

# 모두 train과 동일하게 scaling을 해야함
# scaling 1_1 MinMax
# x_train = x_train/255
# x_test = x_test/255

#2
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# model = Sequential()
# model.add(Conv2D(3, (25,25), input_shape=(100,100,3), activation='sigmoid'))
# model.add(MaxPooling2D())
# model.add(Conv2D(2,(30,30), padding='valid',activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv2D(2,(20,20), padding='valid',activation='relu'))
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(3, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

model.summary()


#3
model.compile(loss = 'binary_crossentropy', optimizer='adam',
              metrics = ['accuracy'])
es = EarlyStopping(monitor='val_loss', mode = 'auto',
                   patience = 50, verbose = 1,
                   restore_best_weights=True)
fit_start = tm.time()
hist = model.fit(x_train, y_train, epochs= 1000, batch_size = 64,
          verbose= 1, validation_data=(x_test,y_test), 
          validation_split=0.2, callbacks=[es])
fit_end = tm.time()
fit_time = np.round(fit_end - fit_start, 2)

#4
loss = model.evaluate(x_test, y_test,verbose=1)
# y_predict = model.predict(x_test)

# acc = accuracy_score(y_test, np.round(y_predict))
# print(y_predict)

print('gen time ', gen_time)
print('fit time', fit_time)
print('loss', loss)
# print('acc', acc)



# gen time  0.42
# fit time 8.71
# loss [0.6921517848968506, 1.0]
# acc 1.0

# gen time  0.42
# fit time 8.71
# loss [0.6931471824645996, 1.0]
# acc 1.0

# gen time  0.41
# fit time 8.37
# loss [0.6921163201332092, 1.0]
# acc 1.0

# gen time  0.44
# fit time 24.51
# loss [0.6907598376274109, 1.0]
# acc 1.0

# gen time  0.47
# fit time 5.7
# loss [0.6928812265396118, 1.0]