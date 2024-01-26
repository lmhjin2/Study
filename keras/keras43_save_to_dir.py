# dir = directory = folder

import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img, to_categorical
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip = True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range= 0.7,
    fill_mode='nearest'
)
augument_size = 40000

randidx = np.random.randint(x_train.shape[0], size = augument_size, )
        # np.random.randint(60000, 40000)   6만개중에 4만개의 숫자를 뽑아내라
# print(randidx) # [38946 26504 25897 ... 19778 49735 50152] list
# print(np.min(randidx), np.max(randidx)) # 4 59999



x_augumented = x_train[randidx].copy()  # 원래 안써도 되는데 가끔 주소 공유 억까가 있어서 .copy로 억까 방지.
# 검색해보면됨. 책에도있음. 어떤책인진 모름
y_augumented = y_train[randidx].copy()

x_augumented = x_augumented.reshape(
    x_augumented.shape[0],
    x_augumented.shape[1],
    x_augumented.shape[2],1
)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented,
    batch_size=augument_size,
    shuffle=False,
    save_to_dir='c:/_data/temp/'        # 이거 한줄 생긴거임.
).next()[0]