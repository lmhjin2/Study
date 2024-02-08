import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,mean_squared_log_error, accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression, ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 1)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits =  5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators

allAlgorithms = all_estimators(type_filter='classifier')

for name, algorithm in allAlgorithms:
    try:
        #2 모델
        model = algorithm()
        #3 훈련
        scores = cross_val_score(model, x_train, y_train, cv = kfold)
        print('============', name, '============')
        print('r2:', scores, "\n 평균 r2:", round(np.mean(scores), 4))
        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
        r2 = r2_score(y_test, y_predict)
        print('cross_val_precit r2:', r2)
    except Exception as e:
        # print(name, '에러', e)
        continue

# r2: 0.19037582699251843
# RMSE: 62.21322867804605
# RMSLE: 0.42893657582040895

# ElasticNetCV 의 정답률: 0.26655983401648775

# acc: [0.49154706 0.4308831  0.36568874 0.42604099 0.39912763 0.49807795
#  0.46417987 0.45828219 0.37906082 0.36389147]
#  평균 acc: 0.4277

# acc: [0.39761427 0.50790494 0.41064636 0.43083409 0.45718859] 
#  평균 acc: 0.4408

# acc: [0.39761427 0.50790494 0.41064636 0.43083409 0.45718859] 
#  평균 acc: 0.4408


# ============ AdaBoostClassifier ============
# r2: [0.01408451 0.         0.01408451 0.         0.        ]
#  평균 r2: 0.0056
# ============ BaggingClassifier ============
# r2: [0.01408451 0.01408451 0.         0.01428571 0.02857143]
#  평균 r2: 0.0142
# ============ BernoulliNB ============
# r2: [0.01408451 0.01408451 0.04225352 0.01428571 0.        ]
#  평균 r2: 0.0169
# ============ CategoricalNB ============
# r2: [nan nan nan nan nan]
#  평균 r2: nan
# ============ DecisionTreeClassifier ============
# r2: [0.         0.         0.01408451 0.01428571 0.01428571]
#  평균 r2: 0.0085
# ============ DummyClassifier ============
# r2: [0.01408451 0.01408451 0.01408451 0.         0.01428571]
#  평균 r2: 0.0113
# ============ ExtraTreeClassifier ============
# r2: [0.         0.         0.         0.         0.01428571]
#  평균 r2: 0.0029
# ============ ExtraTreesClassifier ============
# r2: [0.01408451 0.02816901 0.         0.01428571 0.01428571]
#  평균 r2: 0.0142
# ============ GaussianNB ============
# r2: [0.01408451 0.         0.         0.         0.        ] 
#  평균 r2: 0.0028
# ============ GaussianProcessClassifier ============
# r2: [0.         0.         0.         0.01428571 0.        ]
#  평균 r2: 0.0029
# ============ GradientBoostingClassifier ============
# r2: [0.01408451 0.01408451 0.         0.02857143 0.01428571]
#  평균 r2: 0.0142
# ============ HistGradientBoostingClassifier ============
# r2: [0.         0.         0.01408451 0.04285714 0.        ]
#  평균 r2: 0.0114
# ============ KNeighborsClassifier ============
# r2: [0. 0. 0. 0. 0.]
#  평균 r2: 0.0
# ============ LabelPropagation ============
# r2: [0.01408451 0.01408451 0.         0.01428571 0.01428571]
#  평균 r2: 0.0113
# ============ LabelSpreading ============
# r2: [0.01408451 0.01408451 0.         0.01428571 0.01428571]
#  평균 r2: 0.0113
# ============ LinearDiscriminantAnalysis ============
# r2: [0.         0.         0.01408451 0.01428571 0.01428571]
#  평균 r2: 0.0085
# ============ LinearSVC ============
# r2: [0.         0.         0.01408451 0.         0.01428571]
#  평균 r2: 0.0057
# ============ LogisticRegression ============
# r2: [0.         0.         0.01408451 0.02857143 0.        ]
#  평균 r2: 0.0085
# ============ MLPClassifier ============
# r2: [0.         0.         0.         0.01428571 0.01428571]
#  평균 r2: 0.0057
# ============ NearestCentroid ============
# r2: [0. 0. 0. 0. 0.]
#  평균 r2: 0.0
# ============ PassiveAggressiveClassifier ============
# r2: [0.01408451 0.         0.         0.         0.        ]
#  평균 r2: 0.0028
# ============ Perceptron ============
# r2: [0.         0.01408451 0.02816901 0.         0.02857143]
#  평균 r2: 0.0142
# ============ RadiusNeighborsClassifier ============
# r2: [ 0.  0.  0. nan  0.]
#  평균 r2: nan
# ============ RandomForestClassifier ============
# r2: [0.         0.01408451 0.         0.02857143 0.01428571]
#  평균 r2: 0.0114
# ============ RidgeClassifier ============
# r2: [0.01408451 0.         0.01408451 0.01428571 0.02857143]
#  평균 r2: 0.0142
# ============ RidgeClassifierCV ============
# r2: [0.         0.         0.01408451 0.02857143 0.01428571]
#  평균 r2: 0.0114
# ============ SGDClassifier ============
# r2: [0.         0.         0.         0.01428571 0.01428571]
#  평균 r2: 0.0057
# ============ SVC ============
# r2: [0. 0. 0. 0. 0.]
#  평균 r2: 0.0
