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
#
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1
train_datagen = ImageDataGenerator(
    rescale=1./255)
test_datagen = ImageDataGenerator(
    rescale=1./255)

path_test = 'c://_data//image//cat_and_dog//Test//'
path_train = 'c://_data//image//cat_and_dog//Train//'

xy_train_data = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),
    batch_size = 10, 
    class_mode = 'binary',
    shuffle=True )

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size = 10, 
    class_mode = 'binary')

print(xy_train_data[0][0].shape) # (160, 100, 100, 1)
print(xy_train_data[0][1].shape) # (160,)

x = []
y = []
failed = []

for i in range(int(20000/100)):
    try:
        xy_data = xy_train_data.next()
        new_x = xy_data[0]
        new_y = xy_data[1]
        if i == 0:
            x = np.array(new_x)
            y = np.array(new_y)
            continue
        x = np.vstack([x, new_x])
        y = np.hstack([y,new_y])
        print("i :", i)
        print(f"{x.shape=}\n{y.shape=}")
    except:
        print("failed i:", i)
print(f"{x.shape=}\n{y.shape=}")



print((xy_train_data.next()))

r = int(np.random.uniform(1,1000))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                        random_state=r, stratify=y)

#2
# model = Sequential
# activation='sigmoid'))

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
hist = model.fit(xy_train_data, 
                epochs= 10,
                verbose= 1, 
                # batch_size = 50,    # fit_generator에선 에러 fit에선 돌아가긴하는데 어차피 안먹힘. 위쪽 batch_size로 조절
                validation_data=xy_test,
                # validation_split = 0.2,  # 못써. 에러.
                steps_per_epoch=16, # 전체 데이터 / batch = 160/10 = 16. 
                # 17은 에러. 15는 배치데이터 손실
                callbacks=[es])

#4
results = model.evaluate(xy_test)

print('loss', results[0])
print('acc', results[1])



# gen time  0.42
# fit time 8.71
# loss [0.6921517848968506, 1.0]
# acc 1.0

# gen time  0.42
# fit time 8.71
# loss [0.6931471824645996, 1.0]
# acc 1.0

# gen time  0.41
# fit time 8.37
# loss [0.6921163201332092, 1.0]
# acc 1.0

# gen time  0.44
# fit time 24.51
# loss [0.6907598376274109, 1.0]
# acc 1.0

# gen time  0.47
# fit time 5.7
