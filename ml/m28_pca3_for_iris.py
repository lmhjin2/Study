# train_test_split -> 스케일링 후 PCA

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
# print(x.shape)  # n_componenets 의 갯수만큼 컬럼이 남음 // (150, n_components)

scaler = StandardScaler()
x = scaler.fit_transform(x)

#2 model
model = RandomForestClassifier(random_state=777)
origin_x = x
for i in range(1,x.shape[1]+1):
    pca = PCA(n_components=i)
    x = pca.fit_transform(origin_x)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0, shuffle=True)
    # param = {'random_state':123}
    # model_list = [RandomForestClassifier()]
    model.fit(x_train, y_train)
    model_score = model.score(x_test,y_test)
    print('n_components:' ,i, "=", model_score)

