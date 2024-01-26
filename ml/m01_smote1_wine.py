import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

datasets = load_wine()
x = datasets.data
y = datasets['target']
# print(x.shape, y.shape) # (178, 13) (178,)
# print(np.unique(y,return_counts=True)) 
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(pd.value_counts(y)) 
# 1    71
# 0    59
# 2    48
# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

    # 앞에서 30개 제거
# x = x[30:]
# y = y[30:]
# print(y)
    # 뒤에서 30개 제거
x = x[:-35]
y = y[:-35]
# print(y)

x_train , x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state = 42, stratify = y 
)

'''
#2
model = Sequential()
model.add(Dense(10, input_shape = (13,)))
model.add(Dense(3, activation='softmax'))

#3 
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience=100, verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es])

#4 평가 예측

results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# print(y_test)       # 원핫 안돼있음
# print(y_predict)    # 원핫 돼있음
y_predict = np.argmax(y_predict, axis=1)
# y_predict = y_predict.tolist() 원래있어야함.
f1 = f1_score(y_test,y_predict, average = 'micro')
# ValueError: average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')
print('loss', results[0])
print('accuracy', results[1])
print('f1', f1)


# loss 0.391865998506546
# accuracy 0.8888888955116272
# f1 0.8888888888888888

# loss 0.2941545248031616
# accuracy 0.8888888955116272
# f1 0.8888888888888888
'''
###################################
print('====================== smote 적용 ==========================')
# 부스팅 3대장? 검색. 이페이지엔 관련된거 1도없음
from imblearn.over_sampling import SMOTE
import sklearn as sk
# print('scikit-learn:', sk.__version__) # 1.1.3
# 타임체크
smote = SMOTE(random_state = 42)    # 갯수 가장 높은녀석 기준으로 나머지를 올림.
x_train, y_train = smote.fit_resample(x_train, y_train)
# print(pd.value_counts(y_train))
# 타임체크.
# 오래걸리면 넘파이로 꼭 저장하기. 한번에 많이 돌리면 시간이 많이많이 늘어남
#2
model = Sequential()
model.add(Dense(10, input_shape = (13,),activation='relu'))
# model.add(Dense(33, acitvation='relu'))
model.add(Dense(3, activation='softmax'))

#3 
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience=100, verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es])

#4 평가 예측

results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# print(y_test)       # 원핫 안돼있음
# print(y_predict)    # 원핫 돼있음
y_predict = np.argmax(y_predict, axis=1)
# y_predict = y_predict.tolist() 원래있어야함.
f1 = f1_score(y_test,y_predict, average = 'micro')
# ValueError: average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')
print('loss', results[0])
print('accuracy', results[1])
print('f1', f1)





