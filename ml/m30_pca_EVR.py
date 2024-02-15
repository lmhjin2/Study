# 스케일링 후 PCA -> train_test_split
# 통상적으로 이렇게함

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
np.set_printoptions(suppress=True)
#1 data
datasets = load_breast_cancer()  # 아래 두개중 아무렇게나 써도됨
x = datasets['data']
y = datasets.target
# print(x.shape)     # (442, 10)    # (569, 30)
#                    # diabetes     # cancer
scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components= 25 )
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=777, shuffle=True)

#2 model
model = RandomForestClassifier()

#3 compile train
model.fit(x_train,y_train)

#4 predict, test
results = model.score(x_test,y_test)
print(x.shape)
print(f"model.score : {results}")

evr = pca.explained_variance_ratio_  # == feature_importances // 높은순으로 정렬됨
print(evr)
# print(sum(evr))     # 1.0 // 모든컬럼(10개) 다 쓸때
evr_cumsum =np.cumsum(evr)
print(evr_cumsum)


evr = pca.explained_variance_ratio_
print(np.cumsum(evr))

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()

# diabetes
# (442, 8)
# model.score : 0.4782412102061392

# cancer
# (569, 25)
# model.score : 0.9298245614035088

