# 스케일링 후 PCA -> train_test_split

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
np.set_printoptions(suppress=True)
#1 data
datasets = load_iris()  # 아래 두개중 아무렇게나 써도됨
x = datasets['data']
y = datasets.target
# print(x.shape, y.shape) # (150, 4) (150)

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=3)
x = pca.fit_transform(x)
# print(x)
# print(x.shape)  # n_componenets 의 갯수만큼 컬럼이 남음 // (150, n_components)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0, shuffle=True, stratify=y)

#2 model
model = RandomForestClassifier()

#3 compile train
model.fit(x_train,y_train)

#4 predict, test
results = model.score(x_test,y_test)
print(x.shape)
print(f"model.score : {results}")
