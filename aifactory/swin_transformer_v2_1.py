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


######################  STV2  #################################################################################
import math
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, initializers
from keras_cv_attention_models.models import register_model
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
    "swin_transformer_v2_base_window12": {"imagenet21k": {192: "0747958b4a891370b8caf538b0c6cf1f"}},
    "swin_transformer_v2_base_window16": {"imagenet": {256: "ac5c965069f4452da28169955e8b0444"}, "imagenet22k": {256: "059b9cc52d036b329345a46c82ee9077"}},
    "swin_transformer_v2_base_window24": {"imagenet22k": {384: "809935e83475c252a96dc6107dccd84f"}},
    "swin_transformer_v2_base_window8": {"imagenet": {256: "28ecbbcc6bfb539896bb9ef025df444c"}},
    "swin_transformer_v2_large_window12": {"imagenet21k": {192: "7f3c92eea3295d61e9d5881a7a935da1"}},
    "swin_transformer_v2_large_window16": {"imagenet22k": {256: "8a124dcd6104596bd8fc362f14b87088"}},
    "swin_transformer_v2_large_window24": {"imagenet22k": {384: "c8782f2a1874ca2979c350a8c2c72c39"}},
    "swin_transformer_v2_small_window16": {"imagenet": {256: "89be9ca9b104fb802331120700497bb0"}},
    "swin_transformer_v2_small_window8": {"imagenet": {256: "2736f7ed872130ee59f86e4982d91de0"}},
    "swin_transformer_v2_tiny_window16": {"imagenet": {256: "95d9754a574ff667aad929f1e49f4d1f"}},
    "swin_transformer_v2_tiny_window8": {"imagenet": {256: "97ece5f8d8012d6d40797df063a5f02b"}},
}


# @backend.register_keras_serializable(package="kecam")
class ExpLogitScale(layers.Layer):
    def __init__(self, axis=-1, init_value=math.log(10.0), max_value=math.log(100.0), **kwargs):
        super().__init__(**kwargs)
        self.axis, self.init_value, self.max_value = axis, init_value, max_value

    def build(self, input_shape):
        if self.axis is None:
            weight_shape = (1,)
        elif self.axis == -1 or self.axis == len(input_shape) - 1:
            weight_shape = (input_shape[-1],)
        else:
            weight_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                weight_shape[ii] = input_shape[ii]

        initializer = initializers.constant(self.init_value)
        self.scale = self.add_weight(name="gamma", shape=weight_shape, initializer=initializer, trainable=True)
        # self.__max_value__ = functional.convert_to_tensor(float(math.log(self.max_value)))
        self.__max_value__ = float(self.max_value)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * functional.exp(functional.minimum(self.scale, self.__max_value__))

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "max_value": self.max_value})
        return config


# @backend.register_keras_serializable(package="kecam")
class MlpPairwisePositionalEmbedding(layers.Layer):
    def __init__(self, hidden_dim=512, attn_height=-1, attn_width=-1, pos_scale=-1, use_absolute_pos=False, is_deploy_mode=False, **kwargs):
        # No weight, just need to wrapper a layer, or will not in model structure
        super().__init__(**kwargs)
        self.hidden_dim, self.attn_height, self.attn_width, self.pos_scale = hidden_dim, attn_height, attn_width, pos_scale
        self.use_absolute_pos, self.is_deploy_mode = use_absolute_pos, is_deploy_mode

    def _build_absolute_coords_(self):
        hh, ww = np.meshgrid(range(0, self.height), range(0, self.width), indexing="ij")
        coords = np.stack([hh, ww], axis=-1).astype("float32")
        coords = coords / [self.height // 2, self.width // 2] - 1
        coords = np.reshape(coords, [-1, coords.shape[-1]]) if self.is_compressed else coords
        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("coords", functional.convert_to_tensor(coords, dtype=self.compute_dtype), persistent=False)
        else:
            self.coords = functional.convert_to_tensor(coords, dtype=self.compute_dtype)

    def _build_relative_index_(self):
        hh, ww = np.meshgrid(range(self.height), range(self.width))
        coords = np.stack([hh, ww], axis=-1).astype("float32")  # [15, 12, 2]
        coords_flatten = np.reshape(coords, [-1, 2])  # [180, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [180, 180, 2]
        # relative_coords = tf.reshape(relative_coords, [-1, 2])  # [196 * 196, 2]

        relative_coords_hh = relative_coords[:, :, 0] + self.height - 1
        relative_coords_ww = (relative_coords[:, :, 1] + self.width - 1) * (2 * self.height - 1)
        relative_coords_hhww = np.stack([relative_coords_hh, relative_coords_ww], axis=-1)
        relative_position_index = np.sum(relative_coords_hhww, axis=-1)  # [180, 180]
        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("relative_position_index", functional.convert_to_tensor(relative_position_index, dtype="int64"), persistent=False)
        else:
            self.relative_position_index = functional.convert_to_tensor(relative_position_index, dtype="int64")

    def _build_relative_coords_(self):
        hh, ww = np.meshgrid(range(-self.height + 1, self.height), range(-self.width + 1, self.width), indexing="ij")
        coords = np.stack([hh, ww], axis=-1).astype("float32")
        if self.pos_scale == -1:
            pos_scale = [self.height, self.width]
        else:
            # If pretrined weights are from different input_shape or window_size, pos_scale is previous actually using window_size
            pos_scale = self.pos_scale if isinstance(self.pos_scale, (list, tuple)) else [self.pos_scale, self.pos_scale]
        coords = coords * 8 / [float(pos_scale[0] - 1), float(pos_scale[1] - 1)]  # [23, 29, 2], normalize to -8, 8
        # torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        coords = np.sign(coords) * np.log(1.0 + np.abs(coords)) / (np.log(2.0) * 3.0)
        coords = np.reshape(coords, [-1, 2])  # [23 * 29, 2]
        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("coords", functional.convert_to_tensor(coords, dtype=self.compute_dtype), persistent=False)
        else:
            self.coords = functional.convert_to_tensor(coords, dtype=self.compute_dtype)

    def build(self, input_shape):
        if self.is_deploy_mode:
            self.deploy_bias = self.add_weight(name="deploy_bias", shape=[1, *input_shape[1:]], initializer="zeros", trainable=False)
            super().build(input_shape)
            return

        if self.use_absolute_pos:
            # input_shape: [batch, height, width, channel] or [batch, height * width, channel]
            self.is_compressed = len(input_shape) == 3
            if self.is_compressed:
                self.height = int(float(input_shape[-2]) ** 0.5) if self.attn_height == -1 else self.attn_height
                self.width = input_shape[-2] // self.height
            else:
                self.height, self.width = input_shape[1:-1]

            self._build_absolute_coords_()
            out_shape = [self.hidden_dim, input_shape[-1]]
        else:
            # input_shape: [batch, num_heads, hh * ww, hh * ww]
            height = int(float(input_shape[-2]) ** 0.5) if self.attn_height == -1 else self.attn_height  # hh == ww, e.g. 14
            width = (input_shape[-2] // height) if self.attn_width == -1 else self.attn_width  # hh == ww, e.g. 14
            self.height, self.width, self.num_heads = height, width, input_shape[1]
            padding = input_shape[-2] - height * width
            self.padding = [[padding, 0], [padding, 0], [0, 0]] if padding > 0 else None

            self._build_relative_coords_()
            self._build_relative_index_()
            out_shape = [self.hidden_dim, self.num_heads]

        self.hidden_weight = self.add_weight(name="hidden_weight", initializer="glorot_uniform", shape=[2, self.hidden_dim], trainable=True)
        self.hidden_bias = self.add_weight(name="hidden_bias", initializer="zeros", shape=[self.hidden_dim], trainable=True)
        self.out = self.add_weight(name="out", initializer="glorot_uniform", shape=out_shape, trainable=True)

        self.is_deploy_mode = False
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.is_deploy_mode:
            return inputs + self.deploy_bias

        pos_bias = self.coords @ self.hidden_weight + self.hidden_bias
        pos_bias = functional.relu(pos_bias)
        pos_bias = pos_bias @ self.out

        if not self.use_absolute_pos:
            pos_bias = functional.gather(pos_bias, self.relative_position_index)  # [hh * ww, hh * ww, num_heads]
            pos_bias = functional.sigmoid(pos_bias) * 16.0
            pos_bias = functional.pad(pos_bias, self.padding) if self.padding else pos_bias
            pos_bias = functional.transpose(pos_bias, [2, 0, 1])
        return inputs + functional.expand_dims(pos_bias, 0)

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "hidden_dim": self.hidden_dim,
                "attn_height": self.attn_height,
                "attn_width": self.attn_width,
                "pos_scale": self.pos_scale,
                "use_absolute_pos": self.use_absolute_pos,
                "is_deploy_mode": self.is_deploy_mode,
            }
        )
        return base_config

    def switch_to_deploy(self):
        deploy_bias = self(initializers.zeros()([1, *self.input_shape[1:]]))
        delattr(self, "hidden_weight")
        delattr(self, "hidden_bias")
        delattr(self, "out")

        # add as weights so can be saved to h5 and loaded back
        self.deploy_bias = self.add_weight(name="deploy_bias", shape=deploy_bias.shape, initializer=initializers.Constant(deploy_bias), trainable=False)
        self.is_deploy_mode = True


# @backend.register_keras_serializable(package="kecam")
class WindowAttentionMask(layers.Layer):
    def __init__(self, height, width, window_height, window_width, shift_height=0, shift_width=0, **kwargs):
        # No weight, just need to wrapper a layer, or will meet some error in model saving or loading...
        self.height, self.width, self.window_height, self.window_width = height, width, window_height, window_width
        self.shift_height, self.shift_width = shift_height, shift_width
        self.blocks = (self.height // self.window_height) * (self.width // self.window_width)
        super().__init__(**kwargs)

    def build(self, input_shape):
        hh_split = [0, self.height - self.window_height, self.height - self.shift_height, self.height]
        ww_split = [0, self.width - self.window_width, self.width - self.shift_width, self.width]
        mask_value, total_ww, mask = 0, len(ww_split) - 1, []
        for hh_id in range(len(hh_split) - 1):
            hh = hh_split[hh_id + 1] - hh_split[hh_id]
            rr = [np.zeros([hh, ww_split[id + 1] - ww_split[id]], dtype="float32") + (id + mask_value) for id in range(total_ww)]
            mask.append(np.concatenate(rr, axis=-1))
            mask_value += total_ww
        mask = np.concatenate(mask, axis=0)
        # return mask

        mask = np.reshape(mask, [self.height // self.window_height, self.window_height, self.width // self.window_width, self.window_width])
        mask = np.transpose(mask, [0, 2, 1, 3])
        mask = np.reshape(mask, [-1, self.window_height * self.window_width])
        attn_mask = np.expand_dims(mask, 1) - np.expand_dims(mask, 2)
        attn_mask = np.where(attn_mask != 0, -100, 0)
        attn_mask = np.expand_dims(np.expand_dims(attn_mask, 1), 0)  # expand dims on batch and num_heads

        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("attn_mask", functional.convert_to_tensor(attn_mask, dtype=self.compute_dtype), persistent=False)
        else:
            self.attn_mask = functional.convert_to_tensor(attn_mask, dtype=self.compute_dtype)

        self.num_heads, self.query_blocks = input_shape[1], input_shape[2]
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs: [batch_size * blocks, num_heads, query_blocks, query_blocks]
        # where query_blocks = `window_height * window_width`, blocks = `(height // window_height) * (width // window_width)`
        nn = functional.reshape(inputs, [-1, self.blocks, self.num_heads, self.query_blocks, self.query_blocks])
        nn = nn + self.attn_mask
        return functional.reshape(nn, [-1, self.num_heads, self.query_blocks, self.query_blocks])

    def compute_output_shape(self, input_shape):
        return [None, self.num_heads, self.query_blocks, self.query_blocks]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "window_height": self.window_height,
                "window_width": self.window_width,
                "shift_height": self.shift_height,
                "shift_width": self.shift_width,
            }
        )
        return config


def window_mhsa_with_pair_wise_positional_embedding(
    inputs, num_heads=4, key_dim=0, meta_hidden_dim=512, mask=None, pos_scale=-1, out_bias=True, qv_bias=True, attn_dropout=0, out_dropout=0, name=None
):
    input_channel = inputs.shape[-1]
    height, width = inputs.shape[1:-1]
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    qk_out = key_dim * num_heads

    qkv = functional.reshape(inputs, [-1, height * width, inputs.shape[-1]])
    qkv = layers.Dense(qk_out * 3, use_bias=False, name=name and name + "qkv")(qkv)
    query, key, value = functional.split(qkv, 3, axis=-1)
    if qv_bias:
        query = BiasLayer(name=name and name + "query_bias")(query)
        value = BiasLayer(name=name and name + "value_bias")(value)
    query = functional.transpose(functional.reshape(query, [-1, query.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dim]
    key = functional.transpose(functional.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = functional.transpose(functional.reshape(value, [-1, value.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, vv_dim]

    # cosine attention
    norm_query, norm_key = functional.l2_normalize(query, axis=-1, epsilon=1e-6), functional.l2_normalize(key, axis=-2, epsilon=1e-6)
    attn = functional.matmul(norm_query, norm_key)  # [batch, num_heads, hh * ww, hh * ww]
    attn = ExpLogitScale(axis=1, name=name and name + "scale")(attn)  # axis=1 is head dimension
    attn = MlpPairwisePositionalEmbedding(pos_scale=pos_scale, attn_height=height, name=name and name + "pos_emb")(attn)

    if mask is not None:
        attn = mask(attn)
    attention_scores = layers.Softmax(axis=-1, name=name and name + "attention_scores")(attn)

    if attn_dropout > 0:
        attention_scores = layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    attention_output = functional.matmul(attention_scores, value)
    attention_output = functional.transpose(attention_output, [0, 2, 1, 3])
    attention_output = functional.reshape(attention_output, [-1, height, width, num_heads * key_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    # [batch, hh, ww, num_heads * vv_dim] * [num_heads * vv_dim, out] --> [batch, hh, ww, out]
    attention_output = layers.Dense(qk_out, use_bias=out_bias, name=name and name + "output")(attention_output)
    attention_output = layers.Dropout(out_dropout, name=name and name + "out_drop")(attention_output) if out_dropout > 0 else attention_output
    return attention_output


def shifted_window_attention(inputs, window_size, num_heads=4, shift_size=0, pos_scale=-1, name=None):
    input_channel = inputs.shape[-1]
    window_size = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    window_height = window_size[0] if window_size[0] < inputs.shape[1] else inputs.shape[1]
    window_width = window_size[1] if window_size[1] < inputs.shape[2] else inputs.shape[2]
    shift_size = 0 if (window_height == inputs.shape[1] and window_width == inputs.shape[2]) else shift_size
    should_shift = shift_size > 0

    # window_partition, partition windows, ceil mode padding if not divisible by window_size
    # patch_height, patch_width = inputs.shape[1] // window_height, inputs.shape[2] // window_width
    patch_height, patch_width = int(math.ceil(inputs.shape[1] / window_height)), int(math.ceil(inputs.shape[2] / window_width))
    should_pad_hh, should_pad_ww = patch_height * window_height - inputs.shape[1], patch_width * window_width - inputs.shape[2]
    # print(f">>>> shifted_window_attention {inputs.shape = }, {should_pad_hh = }, {should_pad_ww = }")
    if should_pad_hh or should_pad_ww:
        inputs = functional.pad(inputs, [[0, 0], [0, should_pad_hh], [0, should_pad_ww], [0, 0]])

    if should_shift:
        shift_height, shift_width = int(window_height * shift_size), int(window_width * shift_size)
        # functional.roll is not supported by tflite
        # inputs = functional.roll(inputs, shift=(shift_height * -1, shift_width * -1), axis=[1, 2])
        inputs = functional.concat([inputs[:, shift_height:], inputs[:, :shift_height]], axis=1)
        inputs = functional.concat([inputs[:, :, shift_width:], inputs[:, :, :shift_width]], axis=2)

    # print(f">>>> shifted_window_attention {inputs.shape = }, {patch_height = }, {patch_width = }, {window_height = }, {window_width = }")
    # [batch * patch_height, window_height, patch_width, window_width * channels], limit transpose perm <= 4
    nn = functional.reshape(inputs, [-1, window_height, patch_width, window_width * input_channel])
    nn = functional.transpose(nn, [0, 2, 1, 3])  # [batch * patch_height, patch_width, window_height, window_width * channels]
    nn = functional.reshape(nn, [-1, window_height, window_width, input_channel])  # [batch * patch_height * patch_width, window_height, window_width, channels]

    mask = WindowAttentionMask(inputs.shape[1], inputs.shape[2], window_height, window_width, shift_height, shift_width) if should_shift else None
    nn = window_mhsa_with_pair_wise_positional_embedding(nn, num_heads=num_heads, mask=mask, pos_scale=pos_scale, name=name)

    # window_reverse, merge windows
    # [batch * patch_height, patch_width, window_height, window_width * input_channel], limit transpose perm <= 4
    nn = functional.reshape(nn, [-1, patch_width, window_height, window_width * input_channel])
    nn = functional.transpose(nn, [0, 2, 1, 3])  # [batch * patch_height, window_height, patch_width, window_width * input_channel]
    nn = functional.reshape(nn, [-1, patch_height * window_height, patch_width * window_width, input_channel])

    if should_shift:
        # nn = functional.roll(nn, shift=(shift_height, shift_width), axis=[1, 2])
        nn = functional.concat([nn[:, -shift_height:], nn[:, :-shift_height]], axis=1)
        nn = functional.concat([nn[:, :, -shift_width:], nn[:, :, :-shift_width]], axis=2)

    # print(f">>>> shifted_window_attention before: {nn.shape = }, {should_pad_hh = }, {should_pad_ww = }")
    if should_pad_hh or should_pad_ww:
        nn = nn[:, : nn.shape[1] - should_pad_hh, : nn.shape[2] - should_pad_ww, :]  # In case should_pad_hh or should_pad_ww is 0
    # print(f">>>> shifted_window_attention after: {nn.shape = }")

    return nn


def swin_transformer_block(
    inputs, window_size, num_heads=4, shift_size=0, pos_scale=-1, mlp_ratio=4, mlp_drop_rate=0, attn_drop_rate=0, drop_rate=0, name=None
):
    input_channel = inputs.shape[-1]
    attn = shifted_window_attention(inputs, window_size, num_heads, shift_size, pos_scale=pos_scale, name=name + "attn_")
    attn = layer_norm(attn, zero_gamma=True, axis=-1, name=name + "attn_")
    # attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = layers.Add(name=name + "attn_out")([inputs, attn])

    mlp = mlp_block(attn_out, int(input_channel * mlp_ratio), drop_rate=mlp_drop_rate, use_conv=False, activation="gelu", name=name + "mlp_")
    mlp = layer_norm(mlp, zero_gamma=True, axis=-1, name=name + "mlp_")
    # mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "mlp_")
    return layers.Add(name=name + "output")([attn_out, mlp])


def patch_merging(inputs, name=""):
    input_channel = inputs.shape[-1]
    should_pad_hh, should_pad_ww = inputs.shape[1] % 2, inputs.shape[2] % 2
    # print(f">>>> patch_merging {inputs.shape = }, {should_pad_hh = }, {should_pad_ww = }")
    if should_pad_hh or should_pad_ww:
        inputs = functional.pad(inputs, [[0, 0], [0, should_pad_hh], [0, should_pad_ww], [0, 0]])

    # limit transpose perm <= 4
    nn = functional.reshape(inputs, [-1, 2, inputs.shape[2], input_channel])  # [batch * inputs.shape[1] // 2, height 2, inputs.shape[2], input_channel]
    nn = functional.transpose(nn, [0, 2, 1, 3])  # [batch * inputs.shape[1] // 2, inputs.shape[2], height 2, input_channel]
    nn = functional.reshape(nn, [-1, inputs.shape[1] // 2, inputs.shape[2] // 2, 2 * 2 * input_channel])
    nn = layers.Dense(2 * input_channel, use_bias=False, name=name + "dense")(nn)
    nn = layer_norm(nn, axis=-1, name=name)
    return nn


def SwinTransformerV2(
    num_blocks=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    embed_dim=96,
    window_size=7,
    pos_scale=-1,  # If pretrained weights are from different input_shape or window_size, pos_scale is previous actually using window_size
    stem_patch_size=4,
    use_stack_norm=False,  # True for extra layer_norm on each stack end
    extra_norm_period=0,  # > 0 for extra layer_norm frequency in each stack. May combine with use_stack_norm=True
    input_shape=(256, 256, 3),
    num_classes=0,
    drop_connect_rate=0,
    classifier_activation="sigmoid",
    dropout=0,
    pretrained=None,
    model_name="swin_transformer_v2",
    kwargs=None,
):
    """Patch stem"""
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    nn = layers.Conv2D(embed_dim, kernel_size=stem_patch_size, strides=stem_patch_size, use_bias=True, name="stem_conv")(inputs)
    nn = nn if backend.image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)  # channels_first -> channels_last
    nn = layer_norm(nn, axis=-1, name="stem_")
    window_size = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, num_head) in enumerate(zip(num_blocks, num_heads)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            # height, width downsample * 0.5, channel upsample * 2
            nn = patch_merging(nn, name=stack_name + "downsample_")
        cur_pos_scale = pos_scale[stack_id] if isinstance(pos_scale, (list, tuple)) else pos_scale
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            shift_size = 0 if block_id % 2 == 0 else 0.5
            nn = swin_transformer_block(nn, window_size, num_head, shift_size, cur_pos_scale, drop_rate=block_drop_rate, name=block_name)
            global_block_id += 1
            if extra_norm_period > 0 and (block_id + 1) % extra_norm_period == 0 and not (use_stack_norm and block_id == num_block - 1):
                nn = layer_norm(nn, axis=-1, name=block_name + "output_")
        if use_stack_norm and stack_id != len(num_blocks) - 1:  # Exclude last stack
            nn = layer_norm(nn, axis=-1, name=stack_name + "output_")
    nn = layer_norm(nn, axis=-1, name="pre_output_")
    nn = nn if backend.image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_last -> channels_first

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = models.Model(inputs, nn, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "swin_transformer_v2", pretrained)

    add_pre_post_process(model, rescale_mode="torch")
    model.switch_to_deploy = lambda: switch_to_deploy(model)
    return model


def switch_to_deploy(model):
    from keras_cv_attention_models.model_surgery.model_surgery import convert_layers_to_deploy_inplace

    new_model = convert_layers_to_deploy_inplace(model)
    add_pre_post_process(new_model, rescale_mode=model.preprocess_input.rescale_mode, post_process=model.decode_predictions)
    return new_model

@register_model
def SwinTransformerV2Tiny_window8(input_shape=(1,256, 256, 3), num_classes=0, classifier_activation="sigmoid", pretrained="imagenet", **kwargs):
    window_size = kwargs.pop("window_size", 8)
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_tiny_window8", **kwargs)

model = SwinTransformerV2Tiny_window8()
######################  STV2  #################################################################################
# model.summary()

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
train_meta = pd.read_csv('c:/Study/aifactory/dataset/train_meta.csv')
test_meta = pd.read_csv('c:/Study/aifactory/dataset/test_meta.csv')

# 저장 이름
save_name = 'STV2'

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 10 # 훈련 epoch 지정
BATCH_SIZE = 8 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'STV2' # 모델 이름
RANDOM_STATE = 47 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'c:/Study/aifactory/dataset/train_img/'
MASKS_PATH = 'c:/Study/aifactory/dataset/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = 'c:/Study/aifactory/train_output/'
WORKERS = 8    # 원래 4 // (코어 / 2 ~ 코어) 

# 조기종료
EARLY_STOP_PATIENCE = 5

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 1
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format(MODEL_NAME, save_name)

# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights.h5'.format(MODEL_NAME, save_name)

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
print(len(x_tr), len(x_val))

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
# model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['acc'])
# model.summary()

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
# model.summary()

model.load_weights('c:/Study/aifactory/train_output/STV2_final_weights.h5')

"""## 제출 Predict
- numpy astype uint8로 지정
- 반드시 pkl로 저장"""

y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'c:/Study/aifactory/dataset/test_img/{i}')
    y_pred = model.predict(np.array([img]), batch_size=1)

    y_pred = np.where(y_pred[0, :, :, 0] > 0.5, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

from datetime import datetime
dt = datetime.now()
joblib.dump(y_pred_dict, f'c:/Study/aifactory/train_output/y_pred_{dt.day}_{dt.hour}_{dt.minute}.pkl')


