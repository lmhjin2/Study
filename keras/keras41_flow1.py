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
# plt.imshow(arr)   색있는 점몇개만 나옴
# plt.show()

img = np.expand_dims(arr, axis = 0)
print(img.shape)    # (1, 150, 150, 3)
# axis = 1 -> (150, 1, 150, 3)
# axis = 2 -> (150, 150, 1, 3)
# axis = 3 -> (150, 150, 3, 1)

###### 여기부터 증폭 ################################################
datagen = ImageDataGenerator(
    # rescale=1./255 # 이걸 쓰면 밑에서 image = batch[0].astype('uint8') 이 줄에서 삑날거임.
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.3,
    zoom_range=0.5,
    shear_range = 20,
    fill_mode='nearest' # 아마도 기본값
)

it = datagen.flow(img,
                  batch_size=1,
                  )


'''
    # 1-1   5장 그림만.
# fig, ax 찾아보기     #  행       열       몰루  
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10,10)) 
# 여러장의 그림을 한번에 볼때 쓰는거
for i in range(5):
    batch = it.next()
    # print(batch)
    print(batch.shape)  # (1, 150, 150, 3)
    image = batch[0].astype('uint8')
    # print(image)
    print(image.shape)  # (150, 150, 3)
    # image = image/255
    ax[i].imshow(image)
    ax[i].axis('off')
print(np.min(batch), np.max(batch)) # (0.0, 232.0)
plt.show()
'''


'''
    # 1-2   10 장 x, y 숫자랑 같이
# fig, ax 찾아보기     #  행       열       몰루  
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10,10)) 
# 여러장의 그림을 한번에 볼때 쓰는거
for i in range(10):
    plt.subplot(2, 5, i+1)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
# print(np.min(batch), np.max(batch)) # (0.0, 232.0)
plt.show()
'''




    # 1-3       10 장 사진만.
# fig, ax 찾아보기     #  행       열       몰루  
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10,10)) 
# 여러장의 그림을 한번에 볼때 쓰는거
for i in range(10):
    batch = it.next()
    image = batch[0].astype('uint8')
    ax.flat[i].imshow(image)
    ax.flat[i].axis('off')
# print(np.min(batch), np.max(batch)) # (0.0, 232.0)
plt.show()


'''
    # 1-4       10 장 사진만.
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10,10)) 

for i in range(2):
    for j in range(5):
        batch = it.next()
        image = batch[0].astype('uint8')
        
        ax[i, j].imshow(image)
        ax[i, j].axis('off')
# print(np.min(batch), np.max(batch)) # (0.0, 232.0)
plt.show()
'''




