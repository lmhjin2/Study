import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img, to_categorical
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip = True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range= 0.7,
    fill_mode='nearest'
)
augument_size = 40000

randidx = np.random.randint(x_train.shape[0], size = augument_size, )
        # np.random.randint(60000, 40000)   6만개중에 4만개의 숫자를 뽑아내라
# print(randidx) # [38946 26504 25897 ... 19778 49735 50152] list
# print(np.min(randidx), np.max(randidx)) # 4 59999



x_augumented = x_train[randidx].copy()  # 원래 안써도 되는데 가끔 주소 공유 억까가 있어서 .copy로 억까 방지.
# 검색해보면됨. 책에도있음. 어떤책인진 모름
y_augumented = y_train[randidx].copy()

x_augumented = x_augumented.reshape(
    x_augumented.shape[0],
    x_augumented.shape[1],
    x_augumented.shape[2],1
)


# print(x_augumented)
# print(x_augumented.shape)   # (40000, 28, 28)
# print(y_augumented)
# print(y_augumented.shape)   # (40000,)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented,
    batch_size=augument_size,
    shuffle=False,
).next()[0]
# print(x_augumented)
# print(x_augumented.shape)   # (40000, 28, 28, 1)


# print(x_train.shape)    # (60000, )
x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1)   

# 합치기. (mergy)도 가능
x_train = np.concatenate((x_train, x_augumented)) 
y_train = np.concatenate((y_train, y_augumented))   
# print(x_train.shape, y_train.shape)     # (100000, 28, 28, 1) (100000,)
# 원래 했던거랑 비교. cifar10, mnist, cifar100, 남여, catdog, rps, horseman,

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2
model = Sequential()
model.add(Conv2D(4, (2,2), input_shape=(28, 28, 1),
                 activation='sigmoid'))
model.add(Conv2D(3, (3,3), padding='valid', activation='relu'))
model.add(Conv2D(2, (12,12), padding='valid', activation='relu'))
model.add(Conv2D(1, (5,5), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience=50, verbose= 1,
                   restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size = 200,
                validation_split=0.2, verbose=1, callbacks=[es])

#4
loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_train = np.argmax(y_train, axis = 1)
y_test = np.argmax(y_test, axis = 1)
y_predict = np.argmax(y_predict, axis = 1)

accuracy = accuracy_score(y_test, y_predict)

print('loss', loss)
print('acc', acc)
print('accuracy', accuracy)


# loss 0.6493751406669617
# accuracy 0.763700008392334







