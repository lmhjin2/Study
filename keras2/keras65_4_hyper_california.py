import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import *

#1 data
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape)  # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=True, random_state = 0 )
# print(x_train.shape, y_train.shape)  # (16512, 8) (16512,)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

#2 model
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, lr = 0.01):
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr)
    elif optimizer == 'adadelta':
        optimizer = Adadelta(learning_rate=lr)
    
    inputs = Input(shape=(8,), name='inputs')
    # 파이썬 = 인터프리터 언어 (순차적으로 가서 계속 x라고 해도 됨)
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['mse'], loss = 'mse')
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]    
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    
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
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
keras_model = KerasRegressor(build_fn=build_model, verbose=1)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=3, n_jobs=16, verbose=1)

fit_params = {'callbacks': [es], 'epochs': 3}

import time as tm
start_time = tm.time()
model.fit(x_train, y_train, **fit_params)
end_time = tm.time()

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)

print(f'걸린시간 : {round(end_time-start_time, 2)}')
print(f'model.best_params_ : {model.best_params_}')
print(f'model.best_estimator_ : {model.best_estimator_}')
print(f'model.best_score_ : {model.best_score_}')  # train데이터 기준
print(f'model.score : {model.score(x_test,y_test)}') # test데이터 기준
print(f'r2_score : {r2_score(y_test, y_predict)}')

# 걸린시간 : 17.74
# model.best_params_ : {'optimizer': 'rmsprop', 'node3': 64, 'node2': 32, 'node1': 64, 'lr': 0.001, 'drop': 0.3, 'batch_size': 100, 'activation': 'selu'}
# model.best_estimator_ : <keras.wrappers.scikit_learn.KerasRegressor object at 0x00000202579ED040>
# model.best_score_ : -0.5723134875297546
# 42/42 [==============================] - 0s 514us/step - loss: 0.5325 - mse: 0.5325
# model.score : -0.5325345396995544
# r2_score : 0.5916004943410353


