import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
tf.random.set_seed(777)
np.random.seed(777)
# print(tf.__version__)
from keras.datasets import cifar10
from keras.applications import VGG16
import time as tm
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg16.trainable = False  # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.summary()

#3
es = EarlyStopping(monitor='val_accuracy', mode = 'auto',
                   patience = 50, verbose = 1,
                   restore_best_weights=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam',
              metrics=['accuracy'])
start_time = tm.time()
model.fit(x_train, y_train, epochs = 1000, batch_size = 1000, 
          verbose = 1 , validation_split = 0.18 , callbacks=[es])
end_time = tm.time()
run_time = round(end_time - start_time, 2)

#4
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)

y_train = np.argmax(y_train, axis = 1)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis = 1)

acc = accuracy_score(y_test, y_predict)

print('run time', run_time)
print('loss', results[0])
print('acc ', results[1], acc)

