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

    # for 문 copied

# =================================================================================================================
    # class mode None

train_datagen = ImageDataGenerator(
    rescale=1./255 )

test_datagen = ImageDataGenerator(
    rescale=1./255 )

path_train = 'c:\\_data\\image\\cat_and_dog\\Train\\'
path_test = 'c:\\_data\\image\\cat_and_dog\\Test\\'

Xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(120, 120)
    , batch_size= 32
    , class_mode='binary'
    , color_mode='rgb' # default
    , shuffle='True'
    #Found 20000 images belonging to 2 classes.
)
print('train data ok')

test = test_datagen.flow_from_directory(
    path_test
    , target_size=(120, 120)
    , batch_size= 32
    , class_mode=None
    , color_mode='rgb' # default
    
)
print('submit data ok')


X = []
y = []

for i in range(len(Xy_train)):
    images, labels = Xy_train.next()
    X.append(images)
    y.append(labels)

# all_images와 all_labels을 numpy 배열로 변환하면 하나의 데이터로 만들어진 것입니다.
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)


submit = []
for i in range(len(test)):
    images = test.next()
    submit.append(images)

submit = np.concatenate(submit, axis=0)

print(submit.shape)

# print(X.shape)  # (19997, 120, 120, 3)
# print(y.shape)  # (19997,)

# np_path = 'c:/_data/_save_npy/'
# np.save(np_path + 'keras39_3_cat_dog_x_np.npy', arr=X)
# np.save(np_path + 'keras39_3_cat_dog_y_np.npy', arr=y)
# np.save(np_path + 'kaggle_cat_dog_submission_np.npy', arr=submit)


# =================================================================================================================
    # class_mode='binary',





train_datagen = ImageDataGenerator(
    rescale=1./255 )

test_datagen = ImageDataGenerator(rescale=1./255)      # 평가지표이기 때문에 건드리지마         

path_train = "c:\\_data\\image\\cat_and_dog\\train\\"
path_test = "c:\\_data\\image\\cat_and_dog\\test\\"

xy_train = train_datagen.flow_from_directory(
    path_train, 
    target_size=(100,100),              # 사이즈 조절
    batch_size=100,                      # 160이상을 쓰면 x 통데이터로 가져올 수 있다 , 몇 장씩 수치화 시킬건지 정해준다           
    class_mode='binary',
    shuffle=True)

X = []
y = []

for i in range(len(xy_train)):
    batch = xy_train.next()
    X.append(batch[0])          # 현재 배치의 이미지 데이터
    y.append(batch[1])          # 현재 배치의 라벨 데이터
X = np.concatenate(X, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌
y = np.concatenate(y, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌
    


xy_test = test_datagen.flow_from_directory(
    path_test, 
    target_size=(100,100),              # 사이즈 조절
    batch_size=100,                       
    class_mode='binary')

test=[]


for i in range(len(xy_test)):
    batch = xy_test.next()
    test.append(batch[0])          # 현재 배치의 이미지 데이터
                                 # 현재 배치의 라벨 데이터
test = np.concatenate(test, axis=0)   # 리스트에 저장된 여러개의 NUMPY 배열들을 행을 따라 연결하여 하나의 큰 배열을 만들어줌


np_path = "c:\\_data\\_save_npy\\"

np.save(np_path + 'keras37_3_X_train.npy', arr=X)              
np.save(np_path + 'keras37_3_y_train.npy', arr=y)           
np.save(np_path + 'keras37_3_test1.npy', arr=test)   


