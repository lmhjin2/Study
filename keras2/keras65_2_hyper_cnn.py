import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.optimizers import *

#1 data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)    # (60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
es = EarlyStopping(monitor='acc', patience=10, restore_best_weights=True)

#2 model
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, lr = 0.001):
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr)
    elif optimizer == 'adadelta':
        optimizer = Adadelta(learning_rate=lr)
    
    inputs = Input(shape=(28, 28, 1), name='inputs')
    # 파이썬 = 인터프리터 언어 (순차적으로 가서 계속 x라고 해도 됨)
    x = Conv2D(node1, kernel_size=(2,2), activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(node2, kernel_size=(2,2), activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Conv2D(node3, kernel_size=(2,2), activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    
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

import time as tm
start_time = tm.time()
model.fit(x_train, y_train, epochs=3)
end_time = tm.time()

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

print(f'걸린시간 : {round(end_time-start_time, 2)}')
print(f'model.best_params_ : {model.best_params_}')
print(f'model.best_estimator_ : {model.best_estimator_}')
print(f'model.best_score_ : {model.best_score_}')  # train데이터 기준
print(f'model.score : {model.score(x_test,y_test)}') # test데이터 기준
print(f'acc_score : {accuracy_score(y_test, y_predict)}')

# 걸린시간 : 91.68
# model.best_params_ : {'optimizer': 'adam', 'node3': 64, 'node2': 64, 'node1': 32, 'drop': 0.3, 'batch_size': 100, 'activation': 'relu'}    
# model.best_estimator_ : <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001AA38978490>
# model.best_score_ : 0.9735499918460846
# 100/100 [==============================] - 0s 1ms/step - loss: 0.0422 - acc: 0.9853
# model.score : 0.9853000044822693
# acc_score : 0.9853

