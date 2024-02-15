import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import time as tm
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1 데이터
x, y = load_digits(return_X_y=True)
# print(x.shape, y.shape)     # 64 columns
scaler = StandardScaler()
x = scaler.fit_transform(x)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()  # n_components = 9
x = lda.fit_transform(x,y)
print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

#2 model
model = RandomForestClassifier()

#3 compile train
model.fit(x_train,y_train)

#4 predict, test
results = model.score(x_test,y_test)
# print(x.shape)
print(f"model.score : {results}")

# model.score : 0.9583333333333334