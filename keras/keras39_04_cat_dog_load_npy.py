# https://www.kaggle.com/competitions/cat-and-dog-classification-harper2022/overview

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
import os

#1
np_path = 'c:/_data/_save_npy/'
path = 'c:/_data/kaggle/cat_and_dog/'
img_path = 'c:/_data/image/cat_and_dog/'
x = np.load(np_path + 'keras39_3_x_train.npy')
y = np.load(np_path + 'keras39_3_y_train.npy')
test = np.load(np_path + 'keras39_3_test.npy')
# y_test = np.load(np_path + 'keras39_3_y_test.npy')

test_datagen = ImageDataGenerator(
    rescale=1./255 )

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=0)


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
fit_start = tm.time()
hist = model.fit(x_train, y_train, epochs= 300, batch_size = 64,
          verbose= 1, validation_split=0.2, callbacks=[es])
fit_end = tm.time()
fit_time = np.round(fit_end - fit_start, 2)

#4

# test = np.load(path_test) # ImageDataGenerator로 읽어오기
loss = model.evaluate(x_test, y_test, verbose=1)
y_predict = model.predict(test)

print(y_predict.shape)
y_predict = np.around(y_predict.reshape(-1))
print(y_predict)

folder_dir = img_path + 'Test/test'
id_list = os.listdir(folder_dir)

for i, id in enumerate(id_list):
    id_list[i] = int(id.split('.')[0])

for id in id_list:
    print(id)


y_submit = pd.DataFrame({'id':id_list,'Target':y_predict})
print(y_submit)
y_submit.to_csv(path+"submit_0125.csv",index=False)



# submit_df = pd.DataFrame(submit, columns=['Class'])
# submit_df.to_csv('c:/_data/kaggle/cat_and_dog/')


print('fit time', fit_time)
print('loss', loss)

# fit time 87.31
# loss [0.6893544793128967, 1.0]