import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import time as tm

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate([x_train, x_test], axis=0)     
y = np.concatenate([y_train, y_test], axis=0)      
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

xx = x

parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3,0.001,0.01], "max_depth":[4,5,6]},
     {"n_estimators":[90,100,100], "learning_rate":[0.1,0.001,0.01], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
     {"n_estimators":[90,110], "learning_rate":[0.1,0.001,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], 
      "colsample_bylevel":[0.6,0.7,0.9]}
]

n_comp = [154, 331, 486, 713, 784]

model = RandomizedSearchCV(XGBClassifier(
                            # tree_method='hist',
                            # device = 'cuda',
                            ),parameters,n_jobs=-1,cv=2)
for i in n_comp:
    pca = PCA(n_components=i)
    xx = pca.fit_transform(xx)
    x_train = xx[:60000]
    x_test = xx[60000:]
    #3
    start_time = tm.time()
    model.fit(x_train,y_train)
    end_time = tm.time()
    #4
    acc = model.score(x_test, y_test)
    print(f"====================")
    
    
    print(f"PCA={i}")
    print(f"걸린시간={np.round(end_time-start_time,2)}")
    print(f"ACC={acc}")
    
    xx = pca.inverse_transform(xx)

# ====================
# PCA=154
# 걸린시간=78.64
# ACC=0.9654
# ====================
# PCA=331
# 걸린시간=128.09
# ACC=0.9592
# ====================
# PCA=486
# 걸린시간=403.13
# ACC=0.9625
# ====================
# PCA=713
# 걸린시간=588.42
# ACC=0.9638
# ====================
# PCA=784
# 걸린시간=795.61
# ACC=0.9674
# ====================

