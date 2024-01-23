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


gen_start = tm.time()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode = 'nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255)

# dog = 'C:\_data\image\cat_and_dog\Train\Dog'
# cat = 'C:\_data\image\cat_and_dog\Train\Cat'


path_test = 'c://_data//image//cat_and_dog//Test//'
path_train = 'c://_data//image//cat_and_dog//Train//'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(500,500),
    batch_size = 20000, 
    class_mode = 'binary',
    shuffle=True
)
# xy_train_cat = train_datagen.flow_from_directory(
#     cat,
#     target_size=(500,500),
#     batch_size = 10, 
#     class_mode = 'binary',
#     shuffle=True
# )

# xy_train.samples += xy_train_cat.samples
# xy_train.classes = np.concatenate([xy_train.classes, xy_train_cat.classes])



xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(500,500),
    batch_size = 5000, 
    class_mode = 'binary')

gen_end = tm.time()
gen_time = np.round(gen_end - gen_start, 2)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

# scaling 1_1 MinMax
x_train = x_train/255
x_test = x_test/255

#2
model = Sequential()
model.add(Conv2D(3, (25,25), input_shape=(500,500,3), activation='sigmoid'))
model.add(MaxPooling2D())
model.add(Conv2D(2,(30,30), padding='valid',activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(3, (24,24), padding='valid', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(3,(30,30), padding='valid',activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D())
model.add(Conv2D(1,(36,36), padding='valid',activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(3, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


#3
model.compile(loss = 'binary_crossentropy', optimizer='adam',
              metrics = ['accuracy'])
es = EarlyStopping(monitor='val_loss', mode = 'auto',
                   patience=50, verbose=1,
                   restore_best_weights=True)
fit_start = tm.time()
model.fit(x_train, y_train, epochs= 2000, batch_size = 200,
          verbose= 1, validation_split=0.2, callbacks=[es])
fit_end = tm.time()
fit_time = np.round(fit_end - fit_start, 2)

#4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

acc = accuracy_score(y_test, np.round(y_predict))
print(y_predict)
print('gen time ', gen_time)
print('fit time', fit_time)
print('loss', loss)
print('acc', acc)



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
