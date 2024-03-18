import Base
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

np.random.seed(19)       # 0
random.seed(1)         # 42
tf.random.set_seed(99)   # 7

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


###################################################################################################################
def format_name(prefix, name):
    return (
        ("%s_%s" % (prefix, name)) if prefix is not None and name is not None else None
    )


def inception_scaling(x):
    return (tf.cast(x, tf.float32) - 127.5) * (1 / 127.5)


def simple_scaling(x):
    return tf.cast(x, tf.float32) * (1 / 255.0)


def input_scaling(method="inception"):
    if method == "inception":
        return inception_scaling
    elif method == "simple":
        return simple_scaling
    elif method == None:
        return lambda x: tf.cast(x, tf.float32)


scaled_acts = {
    "relu": lambda x: tf.nn.relu(x) * 1.7139588594436646,
    "relu6": lambda x: tf.nn.relu6(x) * 1.7131484746932983,
    "silu": lambda x: tf.nn.silu(x) * 1.7881293296813965,
    "hswish": lambda x: (x * tf.nn.relu6(x + 3) * 0.16666666666666667)
    * 1.8138962328745718,
    "sigmoid": lambda x: tf.nn.sigmoid(x) * 4.803835391998291,
}


class StochDepth(tf.keras.Model):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    def __init__(self, drop_rate, scale_by_keep=False, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def call(self, x, training):
        if not training:
            return x

        batch_size = tf.shape(x)[0]
        r = tf.random.uniform(shape=[batch_size, 1, 1, 1], dtype=x.dtype)
        keep_prob = 1.0 - self.drop_rate
        binary_tensor = tf.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


class SkipInit(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.skip = self.add_weight(
            name="skip",
            shape=(),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        return x * self.skip


class SkipInitChannelwise(tf.keras.layers.Layer):
    def __init__(self, channels, init_val=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.init_val = init_val
        self.skip = self.add_weight(
            name="skip",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.skip.assign(tf.ones(channels) * init_val)

    def call(self, x):
        return x * self.skip

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        config.update({"init_val": self.init_val})
        return config


class StochDepth(tf.keras.Model):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    def __init__(self, drop_rate, scale_by_keep=False, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def call(self, x, training):
        if not training:
            return x

        batch_size = tf.shape(x)[0]
        r = tf.random.uniform(shape=[batch_size, 1, 1], dtype=x.dtype)
        keep_prob = tf.cast(1.0 - self.drop_rate, dtype=self._compute_dtype)
        binary_tensor = tf.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


class Mlp(tf.keras.layers.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer="gelu",
        drop_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features
        self.act_layer = act_layer
        self.drop_rate = drop_rate

        self.fc1 = tf.keras.layers.Dense(self.hidden_features, name="dense_0")
        self.act = tf.keras.layers.Activation(self.act_layer)
        self.fc2 = tf.keras.layers.Dense(self.out_features, name="dense_1")
        self.drop = tf.keras.layers.Dropout(self.drop_rate)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training)
        x = self.fc2(x)
        x = self.drop(x, training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"in_features": self.in_features})
        config.update({"hidden_features": self.hidden_features})
        config.update({"out_features": self.out_features})
        config.update({"act_layer": self.act_layer})
        config.update({"drop_rate": self.drop_rate})
        return config


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """

    input_shape = tf.shape(x)
    B = input_shape[0]
    H = input_shape[1]
    W = input_shape[2]
    C = input_shape[3]

    x = tf.reshape(
        x, (B, H // window_size, window_size, W // window_size, window_size, C)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, (-1, window_size, window_size, C))
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """

    B = tf.shape(windows)[0] // (H * W // (window_size * window_size))

    x = tf.reshape(
        windows, (B, H // window_size, W // window_size, window_size, window_size, -1)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (B, H, W, -1))
    return x


def log_n(x, n):
    return tf.math.log(x) / tf.math.log(tf.cast(n, x.dtype))


class WindowAttention(tf.keras.layers.Layer):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

        self.logit_max = tf.cast(tf.math.log(1.0 / 0.01), dtype=self._compute_dtype)

        cpb_mlp_dense1 = tf.keras.layers.Dense(
            512,
            use_bias=True,
            name="cpb_mlp/dense_0",
        )
        cpb_mlp_act = tf.keras.layers.Activation("relu", name="cpb_mlp/relu")
        cpb_mlp_dense2 = tf.keras.layers.Dense(
            self.num_heads,
            use_bias=False,
            name="cpb_mlp/dense_1",
        )

        # MLP to generate continuous relative position bias
        self.cpb_mlp = [
            cpb_mlp_dense1,
            cpb_mlp_act,
            cpb_mlp_dense2,
        ]

        relative_coords_h = range(-(self.window_size[0] - 1), self.window_size[0])
        relative_coords_w = range(-(self.window_size[1] - 1), self.window_size[1])

        relative_coords_table = tf.meshgrid(
            relative_coords_h, relative_coords_w, indexing="ij"
        )
        relative_coords_table = tf.stack(relative_coords_table)
        relative_coords_table = tf.transpose(relative_coords_table, (1, 2, 0))
        relative_coords_table = tf.expand_dims(relative_coords_table, axis=0)
        relative_coords_table = tf.cast(relative_coords_table, self.dtype)

        if self.pretrained_window_size[0] > 0:
            relative_coords_table_0 = relative_coords_table[:, :, :, 0] / (
                self.pretrained_window_size[0] - 1
            )
            relative_coords_table_1 = relative_coords_table[:, :, :, 1] / (
                self.pretrained_window_size[1] - 1
            )
        else:
            relative_coords_table_0 = relative_coords_table[:, :, :, 0] / (
                self.window_size[0] - 1
            )
            relative_coords_table_1 = relative_coords_table[:, :, :, 1] / (
                self.window_size[1] - 1
            )
        relative_coords_table = tf.stack(
            [relative_coords_table_0, relative_coords_table_1], axis=3
        )

        relative_coords_table *= 8
        relative_coords_table = (
            tf.math.sign(relative_coords_table)
            * log_n(tf.math.abs(relative_coords_table) + 1.0, 2)
            / np.log2(8)
        )
        self.relative_coords_table = relative_coords_table

        # get pair-wise relative position index for each token inside the window
        coords_h = range(self.window_size[0])
        coords_w = range(self.window_size[1])

        # 2, Wh, Ww
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing="ij"))

        # 2, Wh*Ww
        coords_flatten = tf.reshape(
            coords, (2, self.window_size[0] * self.window_size[1])
        )

        # 2, Wh*Ww, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

        # Wh*Ww, Wh*Ww, 2
        relative_coords = tf.transpose(relative_coords, (1, 2, 0))

        # shift to start from 0
        relative_coords_0 = relative_coords[:, :, 0] + (self.window_size[0] - 1)
        relative_coords_1 = relative_coords[:, :, 1] + (self.window_size[1] - 1)
        relative_coords_0 = relative_coords_0 * (2 * self.window_size[1] - 1)
        relative_coords = tf.stack([relative_coords_0, relative_coords_1], axis=2)

        # Wh*Ww, Wh*Ww
        relative_position_index = tf.math.reduce_sum(relative_coords, axis=-1)
        self.relative_position_index = relative_position_index

        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=False, name="qkv")

        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim, use_bias=True, name="proj")
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def build(self, input_shape):
        logit_init = tf.keras.initializers.Constant(tf.math.log(10.0))
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(self.num_heads, 1, 1),
            initializer=logit_init,
            trainable=True,
        )

        if self.qkv_bias:
            self.q_bias = self.add_weight(
                name="q_bias",
                shape=(self.dim,),
                initializer="zeros",
            )
            self.v_bias = self.add_weight(
                name="v_bias",
                shape=(self.dim,),
                initializer="zeros",
            )
        else:
            self.q_bias = None
            self.v_bias = None

        self.built = True

    def call(self, x, training=None, mask=None):
        input_shape = tf.shape(x)
        B_, N, C = input_shape[0], input_shape[1], input_shape[2]

        qkv = self.qkv(x)
        if self.qkv_bias:
            qkv_bias = tf.concat(
                (
                    self.q_bias,
                    tf.zeros_like(self.q_bias),
                    self.v_bias,
                ),
                axis=0,
            )
            qkv = qkv + qkv_bias

        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, -1))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = tf.math.l2_normalize(q, axis=-1) @ tf.transpose(
            tf.math.l2_normalize(k, axis=-1), (0, 1, 3, 2)
        )
        logit_scale = tf.math.exp(tf.math.minimum(self.logit_scale, self.logit_max))
        attn = attn * logit_scale

        relative_position_bias_table = self.relative_coords_table
        for layer in self.cpb_mlp:
            relative_position_bias_table = layer(relative_position_bias_table)

        relative_position_bias_table = tf.reshape(
            relative_position_bias_table,
            (-1, self.num_heads),
        )

        # Wh*Ww,Wh*Ww,nH
        relative_position_bias = tf.reshape(
            tf.gather(
                relative_position_bias_table,
                tf.reshape(self.relative_position_index, (-1,)),
            ),
            (
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ),
        )

        # nH, Wh*Ww, Wh*Ww
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        relative_position_bias = 16 * tf.math.sigmoid(relative_position_bias)
        attn = attn + tf.expand_dims(relative_position_bias, 0)

        if mask is not None:
            nW = tf.shape(mask)[0]
            mask = tf.expand_dims(tf.expand_dims(mask, 1), 0)
            attn = tf.reshape(attn, (B_ // nW, nW, self.num_heads, N, N)) + mask
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn, training=training)

        x = tf.reshape(tf.transpose(attn @ v, (0, 2, 1, 3)), (B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class SwinTransformerBlock(tf.keras.layers.Layer):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input reslution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (str, optional): Activation layer. Default: "gelu"
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer.  Default: tf.keras.layers.LayerNormalization
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer="gelu",
        norm_layer=tf.keras.layers.LayerNormalization,
        pretrained_window_size=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(epsilon=1e-5, name="norm1")
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=(pretrained_window_size, pretrained_window_size),
            name="attn",
        )

        self.drop_path = (
            StochDepth(drop_path, scale_by_keep=True)
            if drop_path > 0.0
            else tf.keras.layers.Layer()
        )
        self.norm2 = norm_layer(epsilon=1e-5, name="norm2")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop_rate=drop,
            name="mlp",
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            h_sizes = (
                H - self.window_size,
                self.window_size - self.shift_size,
                self.shift_size,
            )
            w_sizes = (
                W - self.window_size,
                self.window_size - self.shift_size,
                self.shift_size,
            )
            cnt = 0
            img_mask = []
            for h in h_sizes:
                img_line = []
                for w in w_sizes:
                    img_line.append(
                        tf.constant(cnt, shape=(h, w), dtype=self._compute_dtype)
                    )
                    cnt += 1
                img_mask.append(tf.concat(img_line, axis=1))

            img_mask = tf.concat(img_mask, axis=0)
            img_mask = tf.expand_dims(tf.expand_dims(img_mask, 0), 3)

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, (-1, self.window_size * self.window_size)
            )
            attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(
                mask_windows, 2
            )
            attn_mask = tf.where(attn_mask != 0, float(-100.0), attn_mask)
            attn_mask = tf.where(attn_mask == 0, float(0.0), attn_mask)
        else:
            attn_mask = None

        self.attn_mask = attn_mask

    def call(self, x, training=None):
        H, W = self.input_resolution
        input_shape = tf.shape(x)
        B = input_shape[0]
        L = input_shape[1]
        C = input_shape[2]

        shortcut = x
        x = tf.reshape(x, (B, H, W, C))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)

        # nW*B, window_size*window_size, C
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size, C))

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = tf.reshape(
            attn_windows, (-1, self.window_size, self.window_size, C)
        )

        # B H' W' C
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2)
            )
        else:
            x = shifted_x
        x = tf.reshape(x, (B, H * W, C))
        x_normed = self.norm1(x)
        x_dropped = self.drop_path(x_normed, training=training)
        x = shortcut + x_dropped

        # FFN
        x_mlp = self.norm2(self.mlp(x, training=training))
        x = x + self.drop_path(x_mlp, training=training)

        return x


class PatchMerging(tf.keras.layers.Layer):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer.  Default: tf.keras.layers.LayerNormalization
    """

    def __init__(
        self,
        input_resolution,
        dim,
        norm_layer=tf.keras.layers.LayerNormalization,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = tf.keras.layers.Dense(
            2 * dim,
            use_bias=False,
            name="reduction",
        )
        self.norm = norm_layer(epsilon=1e-5, name="norm")

    def call(self, x, training=None):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        input_shape = tf.shape(x)
        B = input_shape[0]
        L = input_shape[1]
        C = input_shape[2]

        x = tf.reshape(x, (B, H, W, C))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # B H/2 W/2 4*C
        x = tf.reshape(x, (B, int(H / 2 * W / 2), 4 * C))  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x


def BasicLayer(
    x,
    dim,
    input_resolution,
    depth,
    num_heads,
    window_size,
    name,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop=0.0,
    attn_drop=0.0,
    drop_path=0.0,
    norm_layer=tf.keras.layers.LayerNormalization,
    downsample=None,
    pretrained_window_size=0,
):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer. Default: tf.keras.layers.LayerNormalization
        downsample (tf.keras.layers.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        pretrained_window_size (int): Local window size in pre-training.
    """

    # build blocks
    for i in range(depth):
        x = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if (i % 2 == 0) else window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
            pretrained_window_size=pretrained_window_size,
            name=f"{name}/swin_transformer_block_{i}",
        )(x)

    # patch merging layer
    if downsample is not None:
        x = downsample(
            input_resolution,
            dim=dim,
            norm_layer=norm_layer,
            name=f"{name}/patch_merging",
        )(x)
    return x


def PatchEmbed(x, img_size=224, patch_size=4, embed_dim=96, norm_layer=None):
    r"""Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer. Default: None
    """

    patch_size = (patch_size, patch_size)

    x = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)(x)
    x = tf.keras.layers.Reshape(target_shape=(-1, embed_dim))(x)
    if norm_layer is not None:
        x = norm_layer(epsilon=1e-5)(x)

    return x


def SwinTransformerV2(
    x,
    img_size=224,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    norm_layer=tf.keras.layers.LayerNormalization,
    patch_norm=True,
    pretrained_window_sizes=[0, 0, 0, 0],
):
    r"""Swin Transformer
        A TensorFlow impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (tf.keras.layers.Layer): Normalization layer. Default: tf.keras.layers.LayerNormalization.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    num_layers = len(depths)

    patch_resolution = img_size // patch_size
    patches_resolution = [patch_resolution, patch_resolution]

    # stochastic depth
    dpr = [x for x in tf.linspace(float(0.0), drop_path_rate, sum(depths))]

    x = PatchEmbed(
        x,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        norm_layer=norm_layer if patch_norm else None,
    )

    x = tf.keras.layers.Dropout(rate=drop_rate)(x)

    # build layers
    for i_layer in range(num_layers):
        x = BasicLayer(
            x,
            dim=int(embed_dim * 2**i_layer),
            input_resolution=(
                patches_resolution[0] // (2**i_layer),
                patches_resolution[1] // (2**i_layer),
            ),
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging if (i_layer < num_layers - 1) else None,
            pretrained_window_size=pretrained_window_sizes[i_layer],
            name=f"swin_body_{i_layer}",
        )

    x = norm_layer(epsilon=1e-5, name="predictions_norm")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="predictions_globalavgpooling")(x)

    if num_classes > 0:
        x = tf.keras.layers.Dense(num_classes, name="predictions_dense")(x)
    return x


definitions = {
    "Base": {
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
    }
}


def SwinV2(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="Base",
    input_scaling="inception",
    window_size=10,
    stochdepth_rate=0.2,
    **kwargs,
):
    definition = definitions[definition_name]
    embed_dim = definition["embed_dim"]
    depths = definition["depths"]
    num_heads = definition["num_heads"]

    img_input = tf.keras.layers.Input(shape=in_shape)
    x = Base.input_scaling(method=input_scaling)(img_input)

    x = SwinTransformerV2(
        x,
        img_size=in_shape[0],
        in_chans=in_shape[2],
        num_classes=out_classes,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        drop_path_rate=stochdepth_rate,
        **kwargs,
    )

    x = tf.keras.layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = tf.keras.Model(img_input, x, name=f"SwinV2-{definition_name}")
    return model
model = SwinV2()
###################################################################################################################
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

"""&nbsp;

## parameter 설정
"""

# 사용할 데이터의 meta정보 가져오기

train_meta = pd.read_csv('d:/data/aispark/dataset/train_meta.csv')
test_meta = pd.read_csv('d:/data/aispark/dataset/test_meta.csv')


# 저장 이름
save_name = 'base_line'

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 100 # 훈련 epoch 지정
BATCH_SIZE = 8 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'unet' # 모델 이름
RANDOM_STATE = 47 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'd:/data/aispark/dataset/train_img/'
MASKS_PATH = 'd:/data/aispark/dataset/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = 'c:/Study/aifactory/train_output/'
WORKERS = 16    # 원래 4 // (코어 / 2 ~ 코어) 

# 조기종료
EARLY_STOP_PATIENCE = 20

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
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

# model 불러오기
model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['acc'])
model.summary()

# print(np.unique(x_tr.shape,return_counts=True))
# print(np.unique(x_val.shape,return_counts=True))


# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE,restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='loss', verbose=1,
save_best_only=True, mode='auto', period=CHECKPOINT_PERIOD)
# rlr
rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto', patience=10, verbose=1, factor=0.5)
"""&nbsp;

## model 훈련
"""

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

"""&nbsp;

## model save
"""

print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))

"""## inference

- 학습한 모델 불러오기
"""

model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

# model.load_weights('c:/Study/aifactory/train_output/model_unet_base_line_final_weights.h5')

"""## 제출 Predict
- numpy astype uint8로 지정
- 반드시 pkl로 저장

"""

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


