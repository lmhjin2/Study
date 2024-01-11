import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


#1
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)   # (150,4) 4개의 컬럼과 3개의 class = label 이있음
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (150, 4) (150,)
# print(y)
# print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
# print(pd.value_counts(y))
# 0    50   # 1    50   # 2    50

'''
## keras one-hot encoding ##
from keras.utils import to_categorical
y_encoded = to_categorical(y)
# 마지막 Dense(Layer)에 3 
'''

'''
## pandas one-hot encoding ##
# y_encoded = pd.get_dummies(y)
# 마지막 layer에 3, boolian형 true 1 / false 0 
'''

'''
## sklearn ## 내가한거
from sklearn.preprocessing import OneHotEncoder
category = np.array([datasets.target_names[x] for x in datasets.target])
category = category.reshape(-3, 1)
print(category)
encoder = OneHotEncoder(sparse=False).fit(category)
y_encoded = encoder.transform(category)
print(y_encoded)
'''
'''
## sklearn2 ## 다른사람들
from sklearn.preprocessing import OneHotEncoder
# y_ohe3 = y.reshape(-3, 1)   # 반장의 신기술. 잊어먹어라
y_ohe3 = y.reshape(-1, 1)   # 원래는 이렇게 많이씀. (-1,1)을 쓰면 벡터가 행렬로 바뀜
# y_ohe3 = y.reshape(150, 1)  # 150개 인걸 알아서 하는거임
print(y_ohe3.shape)    # (150, 1) // 벡터를 행렬로 바꾼거. sklearn은 행렬이어야함.
enc = OneHotEncoder(sparse=False).fit(y_ohe3)
y_ohe3 = enc.transform(y_ohe3)
print(y_ohe3)
'''
# y = y.reshape(50,3)  # 스칼라의 개수가 맞아서 ㄱㅊ음. (50,4) 이러면 에러남
# print(y.shape)  # (50,3) # reshape가 순서를 바꾸진 않아서 막 써도 ㄱㅊ음. 내용, 순서, 전체갯수가 바뀌지 않으면 마음대로 바꿔도됨
# y = y.reshape(5, 10, 3), y = y.reshape(30, 5) 전부가능.

## sklearn3 ## 선생님
'''from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y_ohe3 = ohe.fit(y)
y_ohe3 = ohe.transform(y)
'''
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1, 1)
y_ohe3 = ohe.fit_transform(y).toarray()
# print(y_ohe3)
# print(y_ohe3.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x, y_ohe3, stratify= y, train_size = 0.8, random_state = 0 )

print(y_test)
print(np.unique(y_test, return_counts=True))

#2
model = Sequential()
model.add(Dense(120, input_dim = 4))
model.add(Dense(90))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

#3
model.compile(loss = 'categorical_crossentropy',
              optimizer='adam', metrics=['accuracy']) # accuracy == acc
es = EarlyStopping(monitor= 'val_loss', mode = 'min',
                   patience= 30, verbose=1,
                   restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs = 10,
                 batch_size = 96, validation_split=0.2,
                 verbose = 1, callbacks=[es])

#4
results = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
print('loss:', results[0])
print('acc:', results[1])

print(y_predict)
print(y_test)
print(y_predict.shape, y_test.shape)    # (30, 3), (30, 3)

# 이렇게 하면 망함
# y_test = y_test.reshape(90,)      
# y_predict = y_predict.reshape(-1,)
# print(y_test.shape, y_predict.shape)    # (90,), (90,)

y_test = np.argmax(y_test, axis = 1)    # onehot은 axis = 1 이라고 외우면됨
y_predict = np.argmax(y_predict, axis = 1)
print(y_test)       # [0 1 0 2 0 1 2 0 0 1 2 1 1 2 1 2 2 1 1 0 0 2 2 2 0 1 1 2 0 0]
print(y_test.shape) # (30,)
print(y_predict)       # [0 1 0 1 0 1 2 0 0 1 2 2 2 2 1 2 2 0 1 0 0 2 2 2 0 2 2 2 0 0]
print(y_predict.shape) # (30,)

acc = accuracy_score(y_predict, y_test)
print("accuracy_score :", acc)

# 분류에서 acc를 쓰러면 onehot 한거랑 predict 한걸 np.argmax를 통과시켜라. axis = 1








