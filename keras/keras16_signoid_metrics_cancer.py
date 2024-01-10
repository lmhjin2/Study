import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)   # (569, 30)
# print(datasets.feature_names)
x = datasets.data
y = datasets.target
# numpy
print(np.unique(y)) # [ 0 1 ]  

print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))

# pandas 넷다똑같음
# print(pd.DataFrame(y).value_counts())
# print(pd.Series(y).value_counts())
# print(pd.Series.value_counts(y))
print(pd.value_counts(y))

# 1    357
# 0    212


# print(x.shape, y.shape) # (569, 30) (569,)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.85, random_state = 1 )

#2 
model = Sequential()
model.add(Dense(110, input_dim = 30))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1, activation = 'sigmoid'))

#3
model.compile(loss = 'binary_crossentropy', optimizer='adam',
              metrics=['accuracy','mse','mae'])
es = EarlyStopping(monitor = 'val_loss', mode = 'auto',
                   patience = 100, verbose = 1,
                   restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs = 1000,
                 batch_size = 105, validation_split = 0.13,
                 verbose = 2, callbacks=[es])
# loss: 0.09776605665683746
# r2: 0.5982901288073281

#4 
loss = model.evaluate(x_test, y_test)   # x_test를 넣어서 predict한 값을 y_test 와 비교.
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
# print("======================================================================================")
# print(y_test)
# print("======================================================================================")
# print(y_predict)
# print("======================================================================================")

def ACC(y_test, y_predict):
    return accuracy_score(y_test, y_predict)

acc = ACC(y_test, np.round(y_predict))


print('loss:', loss)
print('acc: ', acc)
print('r2:', r2)

# loss: [0.20226573944091797, 0.9534883499145508, 0.046718303114175797, 0.06178978830575943]
# r2: 0.8080396679605841



