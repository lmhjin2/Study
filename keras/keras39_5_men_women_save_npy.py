# https://www.kaggle.com/playlist/men-women-classification

# C:\_data\kaggle\men_women

import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time as tm

# 1
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

path_train = 'c:/_data/kaggle/men_women/train/'
path_test = 'c:/_data/kaggle/men_women/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size = (200,200),
    batch_size = 3330,
    class_mode = 'binary',
    shuffle=True )

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size = (200,200),
    batch_size= 3330,
    class_mode='binary' )


# print(xy_train[0][0].shape) # (3309, 200, 200, 3)
# print(xy_train[0][1].shape) # (3309,)

np_path = 'c:/_data/_save_npy/'
np.save(np_path + 'keras39_5_x_train.npy', arr = xy_train[0][0])
np.save(np_path + 'keras39_5_y_train.npy', arr = xy_train[0][1])
np.save(np_path + 'keras39_5_x_test.npy', arr = xy_test[0][0])
np.save(np_path + 'keras39_5_y_test.npy', arr = xy_test[0][1])








