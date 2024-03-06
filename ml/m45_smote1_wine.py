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

x = x[:-35]
y = y[:-35]

x_train , x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state = 42, stratify = y )
###################################
print('====================== smote 적용 ==========================')
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 42)    # 갯수 가장 높은녀석 기준으로 나머지를 올림.
x_train, y_train = smote.fit_resample(x_train, y_train)
# print(pd.value_counts(y_train))
# 오래걸리면 넘파이로 꼭 저장하기. 한번에 많이 돌리면 시간이 많이많이 늘어남
#2
model = Sequential()
model.add(Dense(10, input_shape = (13,),activation='relu'))
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





