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

model = Sequential()
model.add(Dense(128, input_shape=(9,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))
#3
es = EarlyStopping(monitor='val_loss', mode = 'auto', patience=50, verbose=1, restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = tm.time()
model.fit(x_train,y_train, epochs=500, batch_size=3000, validation_split=0.2, verbose=0, callbacks=[es])
end_time = tm.time()
#4
results = model.evaluate(x_test, y_test)
print(f"====================")
print(f"걸린시간={np.round(end_time-start_time,2)}초")
print(f"ACC={results[1]}")
print(f"Loss={results[0]}")

# ====================
# 걸린시간=5.34초
# ACC=0.9240999817848206
# Loss=0.24516892433166504
