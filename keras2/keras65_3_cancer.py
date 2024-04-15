import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

#1 data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape)  # (569, 30) , (569,)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=True, stratify=y, random_state = 0 )
# print(x_train.shape, y_train.shape)  # (455, 30)

es = EarlyStopping(monitor='acc', patience=10, restore_best_weights=True)

#2 model
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, lr = 0.01):
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr)
    elif optimizer == 'adadelta':
        optimizer = Adadelta(learning_rate=lr)
    
    inputs = Input(shape=(30,), name='inputs')
    # 파이썬 = 인터프리터 언어 (순차적으로 가서 계속 x라고 해도 됨)
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, activation='sigmoid', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss = 'binary_crossentropy')
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]    
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    learning_rates = [0.001, 0.01, 0.1, 0.0001]
    
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            'lr' : learning_rates}

hyperparameters = create_hyperparameter()
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# model = RandomizedSearchCV(build_model, hyperparameters, cv=2, n_iter=10, n_jobs=-1, verbose=1)
    # TypeError : 랜덤서치가 못알아먹는 파라미터가 있어서 안돌아감. // keras모델 자체를 못알아듣는 상태
from keras.wrappers.scikit_learn import KerasClassifier
keras_model = KerasClassifier(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=3, n_jobs=16, verbose=1)

fit_params = {'callbacks': [es], 'epochs': 3}

import time as tm
start_time = tm.time()
model.fit(x_train, y_train, **fit_params)
end_time = tm.time()

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

print(f'걸린시간 : {round(end_time-start_time, 2)}')
print(f'model.best_params_ : {model.best_params_}')
print(f'model.best_estimator_ : {model.best_estimator_}')
print(f'model.best_score_ : {model.best_score_}')  # train데이터 기준
print(f'model.score : {model.score(x_test,y_test)}') # test데이터 기준
print(f'acc_score : {accuracy_score(y_test, y_predict)}')

# 걸린시간 : 7.68
# model.best_params_ : {'optimizer': 'rmsprop', 'node3': 32, 'node2': 64, 'node1': 32, 'lr': 0.0001, 'drop': 0.4, 'batch_size': 300, 'activation': 'relu'}
# model.best_estimator_ : <keras.wrappers.scikit_learn.KerasClassifier object at 0x00000263BD19A820>
# model.best_score_ : 0.6049346923828125
# model.score : 0.6315789222717285
# acc_score : 0.631578947368421