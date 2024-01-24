import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler,StandardScaler

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 가로(평행) 이동 0.1 == 10%
    height_shift_range=0.1, # 세로(수직) 이동 0.1 == 10%
    rotation_range=5,       # 정해진 각도만큼 이미지를 회전
    zoom_range=1.2,         # 확대or 축소. 지금은 1.2배 확대
    shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇개의 좌표를이동시키는 변환
    fill_mode='nearest',    # 빈자리 생긴곳을 근처 비슷한 색으로 채움     
)

test_datagen = ImageDataGenerator(
    rescale=1./255 )     # 실제 데이터를 맞춰야 하기 때문에 다른건 안함.

path_train = 'c:/_data/image/brain/train/'
path_test = 'c:/_data/image/brain/test/'
# 반복자 형태 데이터? (Iterator)형태. x 와 y 가 합체되어있는 형태.
xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200, 200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)   # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200, 200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale'  # color = 'rgb'
)   # Found 120 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001D7DC01BCD0>
print(xy_train.next())   # iterator 데이터는 .next를 쓰면 첫번째 데이터를 보여줌
print(xy_train[0])  # 배치사이즈 10 이라 사진 10장의데이터와 그 y값 총 10개가 나옴.
# print(xy_train[16]) # 배치사이즈 10 이고 사진이 160개라 이제 없음.
                    # 에러:: 전체데이터/batch_size = 160/10 = 16인데
                    # [16]은 17번째 값을 빼라고하니에러가 난다.
print(xy_train[0][0])   # 첫번째 batch의 x값 / 0의 0번째는x
print(xy_train[0][1])   # 첫번째 batch의 y값 / 0의 1번째는y
print(xy_train[0][0].shape) # (10, 200, 200, 3)

# x나 y만 쭉 뽑고싶으면 사진이 160개인거 아니까 batch 160주고 51번줄, 52번줄 ㄱㄱ
# batch 160 주고 53번줄 쓰면 (160, 200, 200, 3) 나옴.

print(type(xy_train))      # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))   # <class 'tuple'>
print(type(xy_train[0][0]))   # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))   # <class 'numpy.ndarray'>





