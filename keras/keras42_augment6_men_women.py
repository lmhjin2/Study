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

