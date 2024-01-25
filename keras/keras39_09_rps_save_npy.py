import numpy as np
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255 )

path_img = 'c:/_data/image/rps/'

data = datagen.flow_from_directory(
    path_img,
    target_size=(150,150),
    batch_size = 50,
    class_mode = 'categorical',
    shuffle=True )

x = []
y = []

for i in range(len(data)):
    batch = data.next()
    x.append(batch[0])          # 이미지 데이터
    y.append(batch[1])          # 라벨 데이터
x = np.concatenate(x, axis=0)   # 데이터 모으기
y = np.concatenate(y, axis=0)   # 데이터 모으기


np_path = 'c:/_data/_save_npy/'

np.save(np_path + 'keras39_09_x.npy', arr = x)
np.save(np_path + 'keras39_09_y.npy', arr = y)

class_labels = data.class_indices
print(class_labels)
# {'paper': 0, 'rock': 1, 'scissors': 2}
print(x.shape, y.shape) 
# (2520, 150, 150, 3) (2520, 3)
