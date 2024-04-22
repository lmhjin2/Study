import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import *

#1 data  
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)    # (60000, 28, 28)
x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

#2 model
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, lr = 0.001):
    inputs = Input(shape=(28*28), name='inputs')
    # 파이썬 = 인터프리터 언어 (순차적으로 가서 계속 x라고 해도 됨)
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'], loss = 'sparse_categorical_crossentropy')
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300]    
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3}

hyperparameters = create_hyperparameter()
# print(hyperparameters)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# model = RandomizedSearchCV(build_model, hyperparameters, cv=2, n_iter=10, n_jobs=-1, verbose=1)
    # TypeError : 랜덤서치가 못알아먹는 파라미터가 있어서 안돌아감. // keras모델 자체를 못알아듣는 상태
from keras.wrappers.scikit_learn import KerasClassifier
keras_model = KerasClassifier(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=3, n_jobs=-1, verbose=1)

import time as tm
start_time = tm.time()
model.fit(x_train, y_train, epochs=3)
end_time = tm.time()

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

# print('걸린시간 : ', round(end_time - start_time, 2))
# print('model.best_params_ :', model.best_params_)
# print('model.best_estimator_ :', model.best_estimator_)
# print('model.best_score_ :', model.best_score_)
# print('model.score : ', model.score(x_test,y_test))
# print('acc_score : ', accuracy_score(y_test, y_predict))

print(f'걸린시간 : {round(end_time-start_time, 2)}')
print(f'model.best_params_ : {model.best_params_}')
print(f'model.best_estimator_ : {model.best_estimator_}')
print(f'model.best_score_ : {model.best_score_}')  # train데이터 기준
print(f'model.score : {model.score(x_test,y_test)}') # test데이터 기준
print(f'acc_score : {accuracy_score(y_test, y_predict)}')

# 걸린시간 : 24.05
# model.best_params_ : {'optimizer': 'rmsprop', 'node3': 128, 'node2': 64, 'node1': 128, 'drop': 0.5, 'batch_size': 300, 'activation': 'elu'}
# model.best_estimator_ : <keras.wrappers.scikit_learn.KerasClassifier object at 0x000002CFEE8883A0>
# model.best_score_ : 0.9108499884605408
# 34/34 [==============================] - 0s 545us/step - loss: 0.2478 - acc: 0.9260
# model.score : 0.9259999990463257
# acc_score : 0.926

