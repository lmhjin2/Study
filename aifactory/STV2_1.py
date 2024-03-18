import os
import warnings
warnings.filterwarnings("ignore")
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K
import sys
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import threading
import random
import rasterio
import os
import numpy as np
import sys
from sklearn.utils import shuffle as shuffle_lists
from keras.models import *
from keras.layers import *
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
import joblib

###################### 전처리 #################################################################################

np.random.seed(0) 
random.seed(42)           
tf.random.set_seed(7)

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

class threadsafe_iter:
    """
    데이터 불러올떼, 호출 직렬화
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE

    return img

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE

    return img

def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg

@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None, image_mode='10bands'):

    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr

    if image_mode == '762':
        fopen_image = get_img_762bands

    i = 0
    # 데이터 shuffle
    while True:

        if shuffle:
            if random_state is None:
                images_path, masks_path = shuffle_lists(images_path, masks_path)
            else:
                images_path, masks_path = shuffle_lists(images_path, masks_path, random_state= random_state + i)
                i += 1

        for img_path, mask_path in zip(images_path, masks_path):

            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []

###################### 전처리 #################################################################################
BATCH_SIZE = 8 # batch size 지정

######################  STV2  #################################################################################
import math
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, initializers
from keras_cv_attention_models.models import register_model
from timm import create_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate
from keras_cv_attention_models.attention_layers import (
    BiasLayer,
    drop_block,
    layer_norm,
    mlp_block,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights
PRETRAINED_DICT = {
    "swin_transformer_v2_tiny_window8": {"imagenet": {256: "97ece5f8d8012d6d40797df063a5f02b"}}
}
# from keras_cv_attention_models import swin_transformer_v2
# base_model = swin_transformer_v2.SwinTransformerV2Tiny_window8(input_shape=(256,256,3),
#                                                                num_classes=1,
#                                                                pretrained="imagenet", 
#                                                        classifier_activation="sigmoid",
#                                                        )

def create_swin_transformer_unet(input_size=(256, 256, 3), num_classes=1):
    # Swin Transformer 모델 로드
    swin_transformer = create_model(
        "swinv2_base_window8_256",
        pretrained=True,
        num_classes=0,  # Fully-connected 층을 제거
        img_size=input_size[0]
    )

    # Swin Transformer의 출력을 사용하여 U-Net 구조를 생성
    base_model = tf.keras.Model(inputs=swin_transformer.input, outputs=swin_transformer.layers[-2].output)

    # U-Net의 Decoder 부분
    start_neurons = 64

    conv4 = base_model.output
    conv4 = UpSampling2D()(conv4)
    c4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    
    c4 = UpSampling2D()(c4)
    c4 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(c4)

    c4 = UpSampling2D()(c4)
    c4 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(c4)

    c4 = UpSampling2D()(c4)
    c4 = Conv2D(start_neurons, (3, 3), activation="relu", padding="same")(c4)

    # 최종 세그멘테이션 맵
    output = Conv2D(num_classes, (1, 1), activation='sigmoid')(c4)

    model = Model(inputs=base_model.input, outputs=output)

    return model

# model = base_model
######################  STV2  #################################################################################

###################### 전처리 #################################################################################

# 두 샘플 간의 유사성 metric
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

# 픽셀 정확도를 계산 metric
def pixel_accuracy (y_true, y_pred):
    sum_n = np.sum(np.logical_and(y_pred, y_true))
    sum_t = np.sum(y_true)

    if (sum_t == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n / sum_t
    return pixel_accuracy

# 사용할 데이터의 meta정보 가져오기
train_meta = pd.read_csv('d:/data/aispark/dataset/train_meta.csv')
test_meta = pd.read_csv('d:/data/aispark/dataset/test_meta.csv')

# 저장 이름
save_name = 'ST'

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 10 # 훈련 epoch 지정
BATCH_SIZE = BATCH_SIZE # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'ST' # 모델 이름
RANDOM_STATE = 47 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'd:/data/aispark/dataset/train_img/'
MASKS_PATH = 'd:/data/aispark/dataset/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = 'c:/Study/aifactory/train_output/'
WORKERS = 8    # 원래 4 // (코어 / 2 ~ 코어) 

# 조기종료
EARLY_STOP_PATIENCE = 5

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format(MODEL_NAME, save_name)

# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_ST.h5'.format(MODEL_NAME, save_name)

# 사용할 GPU 이름
CUDA_DEVICE = 0


# 저장 폴더 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

try:
    np.random.bit_generator = np.random._bit_generator
except:
    pass


# train : val = 8 : 2 나누기
x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)
# print(len(x_tr), len(x_val))

# train : val 지정 및 generator
images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")

from sklearn.metrics import f1_score
def my_f1(y_true,y_pred):
    score = tf.py_function(func=f1_score, inp=[y_true,y_pred], Tout=tf.float32, name='f1_score')
    return score

###################### 전처리 #################################################################################


######################  훈련  #################################################################################

# model 불러오기
model = create_swin_transformer_unet()
model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['acc'])
model.summary()

# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE,restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_loss', verbose=1,
save_best_only=True, mode='auto', period=CHECKPOINT_PERIOD)



print('---model 훈련 시작---')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint, es],
    epochs=EPOCHS,
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH
)
print('---model 훈련 종료---')

## model save
print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))

######################  훈련  #################################################################################

# model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

# model.load_weights('c:/Study/aifactory/train_output/STV2_final_weights.h5')

"""## 제출 Predict
- numpy astype uint8로 지정
- 반드시 pkl로 저장"""

y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'd:/data/aispark/dataset/test_img/{i}')
    y_pred = model.predict(np.array([img]), batch_size=1)

    y_pred = np.where(y_pred[0, :, :, 0] > 0.5, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

from datetime import datetime
dt = datetime.now()
joblib.dump(y_pred_dict, f'c:/Study/aifactory/train_output/y_pred_{dt.day}_{dt.hour}_{dt.minute}.pkl')


