import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.callbacks import EarlyStopping
import time as tm

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate([x_train, x_test], axis=0)    #(70000, 28, 28)
y = np.concatenate([y_train, y_test], axis=0)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
 
scaler = StandardScaler()
x = scaler.fit_transform(x)

lda = LinearDiscriminantAnalysis()  # n_components = 9
x = lda.fit_transform(x,y)
print(x.shape)

x_train = x[:60000]
x_test = x[60000:]

#2
model = XGBClassifier()

#3
start_time = tm.time()
model.fit(x_train,y_train)
end_time = tm.time()
#4
results = model.score(x_test, y_test)
print(f"====================")
print(f"걸린시간={np.round(end_time-start_time,2)}초")
print(f"ACC={results}")

# ====================
# 걸린시간=1.04초
# ACC=0.9211
