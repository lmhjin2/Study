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
    batchs = [100, 200, 300, 400, 500]    
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

model = RandomizedSearchCV(build_model, hyperparameters, cv=2, n_iter=1, n_jobs=-1, verbose=1)






