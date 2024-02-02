from keras.datasets import imdb
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, GRU, Dropout, Flatten, Embedding, Reshape
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# print(x_train.shape, y_train.shape) # (25000,) (25000,)
# print(x_test.shape, y_test.shape)   # (25000,) (25000,)
# print(len(x_train[0]), len(x_test[0])) # 218  68
# print(y_train[:20])  # [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]
# print(np.unique(y_train, return_counts=True)) # [0 1]
#     # (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
    # print(len(np.unique(y_train)))  # 2
# print(max(len(i) for i in x_train)) # 2494
# print(sum(map(len, x_train)) / len(x_train)) # 238.71364

# word_index = imdb.get_word_index()
# print(word_index)   #
# print(len(word_index))  # 88584

x_train = pad_sequences(x_train, padding='pre', maxlen=300, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=300, truncating='pre')

#2
model = Sequential()
model.add(Embedding(input_dim = 10000, output_dim= 12, input_length=300))
model.add(Dropout(0.2))
model.add(Conv1D(22, 2, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(17, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3
model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto', verbose = 1,
                   patience = 100, restore_best_weights=True)
model.fit(x_train,y_train, batch_size = 1000, epochs=1000, verbose=1,
          validation_data=(x_test,y_test), validation_split=0.3, callbacks=[es])


#4
loss, acc = model.evaluate(x_test, y_test)

print('loss',loss)
print('acc',acc)

# loss 1.4752048254013062
# acc 0.8527200222015381

# loss 0.2890586256980896
# acc 0.8789600133895874


