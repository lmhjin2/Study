import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler,StandardScaler
from sklearn.metrics import accuracy_score
import time as tm

# Truncated File 어쩌구 warning 뜰때.

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1

np_path = 'c://_data//_save_npy//'
# np.save(np_path + 'keras39_1_x_train.npy', arr=xy_train_data[0][0])
# np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train_data[0][1])
# np.save(np_path + 'keras39_1_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1])

x_train = np.load(np_path + 'keras39_1_x_train.npy')
y_train = np.load(np_path + 'keras39_1_y_train.npy')
x_test = np.load(np_path + 'keras39_1_x_test.npy')
y_test = np.load(np_path + 'keras39_1_y_test.npy')


# print(x_train)
print(x_train.shape)    # (200, 100, 100, 3)
# print(y_train)
print(y_train.shape)    # (200,)
# print(x_test)
print(x_test.shape)     # (200, 100, 100, 3)
# print(y_test)
print(y_test.shape)     # (200,)


#2
model = Sequential()
model.add(Conv2D(2, (16, 16), input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(2, (12, 12), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(1, (8, 8), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3
model.compile(loss = 'binary_crossentropy', optimizer='adam',
              metrics = ['accuracy'])
es = EarlyStopping(monitor='val_loss', mode = 'auto',
                   patience = 50, verbose = 1,
                   restore_best_weights=True)
hist = model.fit(x_train, y_train, 
                epochs= 10,
                verbose= 1, 
                batch_size = 50,    # fit_generator에선 에러 fit에선 돌아가긴하는데 어차피 안먹힘. 위쪽 batch_size로 조절
                # validation_data=xy_test,
                validation_split = 0.2,  # 못써. 에러.
                # steps_per_epoch=16, # 전체 데이터 / batch = 160/10 = 16. 
                # 17은 에러. 15는 배치데이터 손실
                callbacks=[es])

#4
results = model.evaluate(x_test, y_test,verbose=1)

print('loss', results[0])
print('acc', results[1])


# loss 0.6927167177200317
# acc 1.0

