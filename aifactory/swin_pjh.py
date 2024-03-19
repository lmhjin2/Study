import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from keras.layers import (Input, Conv2D, BatchNormalization, Dropout, Dense, MaxPooling2D, Reshape,
                                     ReLU, Flatten, Softmax, GlobalAveragePooling2D, Concatenate, Add)
from keras import layers, models
from keras.models import Model
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
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
tf.random.set_seed(3)
np.random.seed(3)
random.seed(3)
# import os
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


THESHOLDS = 0.25 
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
    img = np.float32(img) / 65535
    return img

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img
    
def get_mask_arr(path):
    img = rasterio.open(path).read(1)
    img = np.float32(img) / 65535
    img = np.expand_dims(img, axis=-1)  # 마스크에 채널 차원 추가
    return img

def miou(y_true, y_pred, smooth=1e-6):
    # 임계치 기준으로 이진화
    y_pred = tf.cast(y_pred > THESHOLDS, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # mIoU 계산
    iou = (intersection + smooth) / (union + smooth)
    miou = tf.reduce_mean(iou)
    return miou

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

                

input_shape = (256, 256, 3)
patch_size = (4, 4)  # 2-by-2 sized patches
dropout_rate = 0.03  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp =256  # MLP layer size
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
# image_dimension = 224  # Initial image size

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 0.001
# batch_size = 32
# num_epochs = 40
# validation_split = 0.1
# weight_decay = 1e-5
# label_smoothing = 0.1
def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))                                             # [batch_size, height, width, channels]
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))            # [num_windows, window_size, window_size, channels]
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x                                                #  window_partition의 반대 : 최종이미지 형태로 재구성


class DropPath(layers.Layer):                               #  일부 경로를 drop하면서 과적합을 줄여줌 dropout은 노드의 개수를 줄여주지만
    def __init__(self, drop_prob=None, **kwargs):           # 이 친구는 네트워크의 특정 경로를 생략한다  
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output
    
class WindowAttention(layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5             #  attention score 계산시 사용
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):                                   # 레이어가 처음 사용될 때 한 번 호출되어 내부 변수들을 초기화
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1                             # 상대적 위치 bias를 계산하여 저장
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale                      # query에 self.scale을 곱해줌
        k = tf.transpose(k, perm=(0, 1, 3, 2))  # 위에서 나온 결과를 key와 내적
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv
    
class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(SwinTransformer, self).__init__(**kwargs)            
        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):            # 주의 메커니즘에 사용될 상대적 위치 편향과 attention mask를 초기화

        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x
    
class PatchExtract(layers.Layer):                       #  이미지로부터 패치를 추출하는 사용자 정의 층
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]

    def call(self, images):                     # images = [batch_size, height, width, channels]
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",                    # [batch_size, out_height, out_width, patch_dim]
        )
        patch_dim = patches.shape[-1]
        patch_num_x = patches.shape[1]
        patch_num_y = patches.shape[2]
        return tf.reshape(patches, (batch_size, patch_num_x * patch_num_y, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)               #  패치의 원본 내용뿐만 아니라 패치의 상대적 또는 절대적 위치 정보도 학습


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super(PatchMerging, self).__init__()
        self.num_patch = num_patch      # 입력 데이터의 패치 수, 즉 입력 데이터의 height와 width의 패치 수를 튜플 형태
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):                  # x: 입력 텐서. 패치 임베딩들로 구성된 [batch_size, num_patches, channels] 형태
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)
 
 
def swin_transformer_with_mlp_model(input_shape, num_patch, num_patch_x, num_patch_y, embed_dim, num_heads, window_size, shift_size, num_mlp, qkv_bias, dropout_rate):
    input_1 = layers.Input(shape=input_shape, name='img')
    x = PatchExtract(patch_size)(input_1)
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)  
       
    # x = SwinTransformer(                    # 윈도우 이동 x
    #     dim=embed_dim,
    #     num_patch=(num_patch_x, num_patch_y),
    #     num_heads=num_heads,
    #     window_size=window_size,
    #     shift_size=0,
    #     num_mlp=num_mlp,
    #     qkv_bias=qkv_bias,
    #     dropout_rate=dropout_rate,
    # )(x)
    x = SwinTransformer(                    # 윈도우 이동 o
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
    x = Reshape((32, 32, 128))(x)

    # 차원을 증가시키기 위한 Conv2DTranspose 사용
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same', activation='sigmoid')(x)  # 최종 출력

    model = Model(inputs=input_1, outputs=x)
    return model


# input_shape = (256, 256, 3)
# patch_size = (4, 4)  # 2-by-2 sized patches
# dropout_rate = 0.03  # Dropout rate
# num_heads = 8  # Attention heads
# embed_dim = 64  # Embedding dimension
# num_mlp = 8  # MLP layer size
# qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
# window_size = 2  # Size of attention window
# shift_size = 1  # Size of shifting window
# # image_dimension = 224  # Initial image size

# num_patch_x = input_shape[0] // patch_size[0]
# num_patch_y = input_shape[1] // patch_size[1]



# 모델 생성
model = swin_transformer_with_mlp_model(
    input_shape=input_shape, 
    num_patch=num_patch_x * num_patch_y,
    num_patch_x=num_patch_x, 
    num_patch_y=num_patch_y, 
    embed_dim=embed_dim, 
    num_heads=num_heads, 
    window_size=window_size, 
    shift_size=shift_size, 
    num_mlp=num_mlp, 
    qkv_bias=qkv_bias, 
    dropout_rate=dropout_rate
)
# 모델 summary 출력
model.summary()

# input_shape = (256, 256, 3)
# patch_size = (4, 4)  # 2-by-2 sized patches
# dropout_rate = 0.03  # Dropout rate
# num_heads = 8  # Attention heads
# embed_dim = 64  # Embedding dimension
# num_mlp =256  # MLP layer size
# qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
# window_size = 2  # Size of attention window
# shift_size = 1  # Size of shifting window
# # image_dimension = 224  # Initial image size


def get_model(MODEL_NAME, input_height, input_width, n_filters, n_channels):
    # 입력 이미지의 크기와 패치 크기를 바탕으로 num_patch_x와 num_patch_y 계산
    patch_size = (4, 4)  # 여기서는 패치 크기를 (4, 4)로 가정합니다.
    num_patch_x = input_width // patch_size[0]
    num_patch_y = input_height // patch_size[1]

    input_shape = (input_height, input_width, n_channels)  # 입력 이미지의 차원 정의
    num_patch = num_patch_x * num_patch_y  # 전체 패치 수
    embed_dim = 64  # 임베딩 차원
    num_heads = 8  # 주의 메커니즘에서의 헤드 수
    window_size = 2  # 윈도우 크기
    shift_size = 1  # 윈도우 이동 크기
    num_mlp = 256  # MLP의 크기
    qkv_bias = True  # QKV 계산에서 편향 사용 여부
    dropout_rate = 0.03  # 드롭아웃 비율

    if MODEL_NAME == 'swin_transformer':
        model = swin_transformer_with_mlp_model(
        input_shape=input_shape, 
        num_patch=num_patch, 
        num_patch_x=num_patch_x, 
        num_patch_y=num_patch_y, 
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        window_size=window_size, 
        shift_size=shift_size, 
        num_mlp=num_mlp, 
        qkv_bias=qkv_bias, 
        dropout_rate=dropout_rate
    )
        return model
    else:
        raise ValueError(f"Model {MODEL_NAME} is not supported.")
    
# 모델 생성
# model = create_swin_transformerl()
# 모델 summary 출력
# model.summary()

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 1 # 훈련 epoch 지정
BATCH_SIZE = 3 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'swin_transformer' # 모델 이름
RANDOM_STATE = 3 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

lr = 0.001
num_patches = (input_shape[0] // patch_size[0]) * (input_shape[1] // patch_size[1])



 

# 모델 생성 호출 예시
# model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
        
# 예시: 모델 이름으로 모델 생성
# model = get_model('swin_transformer')
model.summary()


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

train_meta = pd.read_csv('c:/Study/aifactory/dataset/train_meta.csv')
test_meta = pd.read_csv('c:/Study/aifactory/dataset/test_meta.csv')


# 저장 이름
save_name = 'base_line'



rlr = ReduceLROnPlateau(monitor='val_miou', patience=10, mode='accuracy', verbose=1, factor=0.5)
# 데이터 위치
IMAGES_PATH = 'c:/Study/aifactory/dataset/train_img/'
MASKS_PATH = 'c:/Study/aifactory/dataset/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = 'c:/Study/aifactory/train_output/'
WORKERS = 20

# 조기종료
EARLY_STOP_PATIENCE = 25

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}_03_19_01.hdf5'.format(MODEL_NAME, save_name)
 
# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_swin_pjh.h5'.format(MODEL_NAME, save_name)

# 사용할 GPU 이름
CUDA_DEVICE = 0

def miou(y_true, y_pred, smooth=1e-6):
    # 임계치 기준으로 이진화
    y_pred = tf.cast(y_pred > THESHOLDS, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # mIoU 계산
    iou = (intersection + smooth) / (union + smooth)
    miou = tf.reduce_mean(iou)
    return miou

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
train_meta = pd.read_csv('c:/Study/aifactory/dataset/train_meta.csv')
x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)

images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]
images_val = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_val = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, BATCH_SIZE, image_mode='762', shuffle=True)
val_generator = generator_from_lists(images_val, masks_val, BATCH_SIZE, image_mode='762', shuffle=True)




# print(images_train)
# print(masks_train)

import segmentation_models as sm
loss = sm.losses.bce_jaccard_loss
# model 불러오기
model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model.compile(optimizer = Adam(learning_rate=lr), loss = loss, metrics = ['accuracy', miou])
model.summary()
# sm.metrics.iou_score

# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_miou', mode='max', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_iou_score', verbose=1,
save_best_only=True, mode='max', period=CHECKPOINT_PERIOD)

print('---model 훈련 시작---')
history = model.fit(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=len(images_val) // BATCH_SIZE,
    callbacks=[checkpoint, es, rlr],
    epochs=EPOCHS,
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH
    
)
# history = model.fit(
#     x=train_generator,  # 훈련 데이터 제너레이터
#     validation_data=val_generator,  # 검증 데이터 제너레이터
#     steps_per_epoch=len(images_train) // BATCH_SIZE,
#     validation_steps=len(images_val) // BATCH_SIZE,
#     epochs=EPOCHS,
#     callbacks=[checkpoint, es, rlr]
# )


print('---model 훈련 종료---')

print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))


# model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
# model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
# model.summary()

# model.load_weights('C:\\_data\\AI factory\\train_output\\model_swin_transformer_base_line_final_weights_03_19_01.h5')


y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'c:/Study/aifactory/dataset/test_img/{i}')
    y_pred = model.predict(np.array([img]), batch_size=1)
    
    y_pred = np.where(y_pred[0, :, :, 0] > 0.25, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

from datetime import datetime
dt = datetime.now()
joblib.dump(y_pred_dict, f'c:/Study/aifactory/train_output/y_pred_{dt.day}_{dt.hour}_{dt.minute}.pkl')


# 한번 잘 엮어봐줘