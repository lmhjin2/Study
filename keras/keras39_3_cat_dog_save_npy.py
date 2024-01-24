import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler,StandardScaler
import time as tm


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_datagen = ImageDataGenerator(
    rescale=1./255 )

test_datagen = ImageDataGenerator(
    rescale=1./255 )

submit_datagen = ImageDataGenerator(
    rescale=1./255 )

path_train = 'c:/_data/image/cat_and_dog/Train/'
path_test = 'c:/_data/image/cat_and_dog/Test/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size = (100,100),
    batch_size = 20000,
    class_mode='binary',
    shuffle=True )

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size = 10000,
    class_mode = 'binary' )


print(xy_train[0][0].shape) # (19997, 100, 100, 3)
print(xy_train[0][1].shape) # (19997,)
print(xy_test[0][0].shape) # (5000, 100, 100, 3)
# print(xy_test[0][1].shape) # (5000,)

np_path = 'c:/_data/_save_npy/'
np.save(np_path + 'keras39_3_x_train.npy', arr = xy_train[0][0])
np.save(np_path + 'keras39_3_y_train.npy', arr = xy_train[0][1])
np.save(np_path + 'keras39_3_test.npy', arr = xy_test[0][0])
# np.save(np_path + 'keras39_3_y_test.npy', arr = xy_test[0][1])



