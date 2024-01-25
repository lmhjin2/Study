import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.preprocessing.image import load_img # 이미지 땡겨와
# from tensorflow.keras.preprocessing.image import img_to_array # 이미지 수치화
from keras.utils import img_to_array, load_img
# 케라스 업데이트 하면서 keras.utils 로 들어옴
import matplotlib.pyplot as plt

# print('텐서', tf.__version__)
# print('파이썬', sys.version)
# 텐서 2.9.0
# 파이썬 3.9.18 (main, Sep 11 2023, 14:09:26) [MSC v.1916 64 bit (AMD64)]


path = 'c:/_data/image/cat_and_dog/Train/Cat/1.jpg'
img = load_img(path,
                 target_size = (150,150), # 주석 먹이면(281, 300, 3)
                                            # 원래 사이즈.(y,x,3) ??
                 )                          # 세로 사진인데 (397, 312, 3)나옴
print(img)
# <PIL.Image.Image image mode=RGB size=150x150 at 0x21795873520>
print(type(img))    # <class 'PIL.Image.Image'>
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr.shape)    # (150, 150, 3)
print(type(arr))    # <class 'numpy.ndarray'>
# plt.imshow(arr)
# plt.show()

img = np.expand_dims(arr, axis = 2)
print(img.shape)    # (1, 150, 150, 3)
# axis = 1 -> (150, 1, 150, 3)
# axis = 2 -> (150, 150, 1, 3)


