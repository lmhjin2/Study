import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator       ###### 이미지를 숫자로 바꿔준다##########
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
import time
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale=1./255 )

test_datagen = ImageDataGenerator(
    rescale=1./255 )

path_img = 'c:/_data/image/horse_human/'

data = train_datagen.flow_from_directory(
    path_img,
    target_size = (300,300),
    batch_size = 100,
    class_mode = 'categorical',
    shuffle=True)

x = []
y = []

for i in range(len(data)):
    batch = data.next()
    x.append(batch[0])          # 현재 배치의 이미지 데이터
    y.append(batch[1])          # 현재 배치의 라벨 데이터
x = np.concatenate(x, axis=0)   # 데이터 모으기
y = np.concatenate(y, axis=0)   # 데이터 모으기


# x_train, x_test, y_train, y_test = train_test_split(x,y,
#             test_size=0.2, random_state=42,stratify=y)

np_path = 'c:/_data/_save_npy/'
#1-1    train_test_split 하고 넘기기
# np.save(np_path + 'keras39_07_x_train.npy', arr = x_train)
# np.save(np_path + 'keras39_07_y_train.npy', arr = y_train)
# np.save(np_path + 'keras39_07_x_test.npy', arr = x_test)
# np.save(np_path + 'keras39_07_y_test.npy', arr = y_test)

#1-2    넘어가서 train_test_split 하기
np.save(np_path + 'keras39_07_x.npy', arr = x)
np.save(np_path + 'keras39_07_y.npy', arr=y)

class_labels = data.class_indices
print(class_labels)
# {'horses': 0, 'humans': 1}

print(x.shape)
# (1027, 300, 300, 3)
print(y.shape)
# (1027, 2)
