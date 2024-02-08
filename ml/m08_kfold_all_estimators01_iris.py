import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import all_estimators

#1 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, shuffle=True, random_state= 0, stratify=y)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2 모델
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
        print('acc:', scores, "\n 평균 acc:", round(np.mean(scores), 4))
        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
        acc = accuracy_score(y_test, y_predict)
        print('cross_val_precit acc:', acc)
    except Exception as e:
        # print(name, '에러', e)
        continue


# acc: [0.93333333 1.         1.         1.         1.         0.93333333
#  0.93333333 1.         1.         0.8       ]
#  평균 acc: 0.96

# acc: [1.         1.         1.         0.93333333 0.93333333 0.93333333
#  1.         1.         0.93333333 0.93333333]
#  평균 acc: 0.9667




# ============ AdaBoostClassifier ============
# acc: [1.         1.         1.         1.         0.66666667 0.66666667
#  0.66666667 1.         1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.95
# ============ BaggingClassifier ============
# acc: [1.         1.         0.66666667 1.         1.         0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9333
# cross_val_precit acc: 0.9666666666666667
# ============ BernoulliNB ============
# acc: [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333 0.33333333
#  0.33333333 0.33333333 0.33333333 0.33333333]
#  평균 acc: 0.3333
# cross_val_precit acc: 0.3333333333333333
# ============ CalibratedClassifierCV ============
# acc: [0.66666667 1.         0.66666667 0.66666667 0.66666667 0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.8333
# cross_val_precit acc: 0.9083333333333333
# ============ CategoricalNB ============
# acc: [0.66666667 0.33333333 0.33333333 0.66666667 0.33333333 0.33333333
#  0.33333333 0.33333333        nan        nan]
#  평균 acc: nan
# ============ ComplementNB ============
# acc: [0.66666667 0.66666667 0.66666667 0.66666667 0.66666667 0.66666667
#  0.66666667 0.66666667 0.66666667 0.66666667]
#  평균 acc: 0.6667
# cross_val_precit acc: 0.6666666666666666
# ============ DecisionTreeClassifier ============
# acc: [1.         1.         1.         1.         0.66666667 0.66666667
#  0.66666667 1.         1.         0.66666667]
#  평균 acc: 0.8667
# cross_val_precit acc: 0.9666666666666667
# ============ DummyClassifier ============
# acc: [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333 0.33333333
#  0.33333333 0.33333333 0.33333333 0.33333333]
#  평균 acc: 0.3333
# cross_val_precit acc: 0.3333333333333333
# ============ ExtraTreeClassifier ============
# acc: [1.         1.         0.66666667 1.         1.         0.66666667
#  0.66666667 1.         1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.925
# ============ ExtraTreesClassifier ============
# acc: [1.         1.         0.66666667 1.         0.66666667 0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.95
# ============ GaussianNB ============
# acc: [1.         1.         0.66666667 1.         1.         0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9333
# cross_val_precit acc: 0.9583333333333334
# ============ GaussianProcessClassifier ============
# acc: [1.         1.         0.66666667 1.         0.66666667 0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.9416666666666667
# ============ GradientBoostingClassifier ============
# acc: [1.         1.         1.         1.         0.66666667 0.66666667
#  0.66666667 1.         1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.9583333333333334
# ============ HistGradientBoostingClassifier ============
# acc: [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333 0.33333333
#  0.33333333 0.33333333 0.33333333 0.33333333]
#  평균 acc: 0.3333
# cross_val_precit acc: 0.9666666666666667
# ============ KNeighborsClassifier ============
# acc: [1.         1.         1.         1.         1.         0.66666667
#  0.66666667 0.66666667 1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.9666666666666667
# ============ LabelPropagation ============
# acc: [1.         1.         1.         1.         1.         0.66666667
#  0.66666667 0.66666667 1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.9583333333333334
# ============ LabelSpreading ============
# acc: [1.         1.         1.         1.         1.         0.66666667
#  0.66666667 0.66666667 1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.9583333333333334
# ============ LinearDiscriminantAnalysis ============
# acc: [1.         1.         1.         1.         0.66666667 0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9333
# cross_val_precit acc: 0.9833333333333333
# ============ LinearSVC ============
# acc: [0.66666667 1.         0.66666667 0.66666667 0.66666667 0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.8333
# cross_val_precit acc: 0.9333333333333333
# ============ LogisticRegression ============
# acc: [1.         1.         0.66666667 1.         0.66666667 0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.9416666666666667
# ============ LogisticRegressionCV ============
# acc: [1.         1.         0.66666667 1.         0.66666667 0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.9833333333333333
# ============ MLPClassifier ============
# acc: [1.         1.         1.         1.         0.66666667 1.
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9667
# cross_val_precit acc: 0.9666666666666667
# ============ MultinomialNB ============
# acc: [1.         1.         1.         1.         0.66666667 1.
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9667
# cross_val_precit acc: 0.9416666666666667
# ============ NearestCentroid ============
# acc: [1.         1.         1.         1.         1.         0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9667
# cross_val_precit acc: 0.9583333333333334
# ============ NuSVC ============
# acc: [1.         1.         1.         1.         1.         0.66666667
#  0.66666667 0.66666667 1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.9583333333333334
# ============ PassiveAggressiveClassifier ============
# acc: [0.66666667 0.66666667 1.         0.66666667 1.         0.33333333
#  0.66666667 0.66666667 0.66666667 1.        ]
#  평균 acc: 0.7333
# cross_val_precit acc: 0.8583333333333333
# ============ Perceptron ============
# acc: [0.66666667 0.66666667 0.66666667 0.66666667 0.33333333 0.66666667
#  0.66666667 0.66666667 0.66666667 0.66666667]
#  평균 acc: 0.6333
# cross_val_precit acc: 0.7833333333333333
# ============ QuadraticDiscriminantAnalysis ============
# acc: [1.         1.         0.66666667 1.         0.66666667 0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.975
# ============ RadiusNeighborsClassifier ============
# acc: [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333 0.33333333
#  0.33333333 0.33333333 0.33333333 0.33333333]
#  평균 acc: 0.3333
# cross_val_precit acc: 0.3333333333333333
# ============ RandomForestClassifier ============
# acc: [1.         1.         0.66666667 1.         0.66666667 0.66666667
#  1.         1.         1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.9666666666666667
# ============ RidgeClassifier ============
# acc: [0.66666667 1.         0.66666667 0.66666667 0.66666667 0.66666667
#  1.         0.66666667 0.66666667 1.        ]
#  평균 acc: 0.7667
# cross_val_precit acc: 0.8666666666666667
# ============ RidgeClassifierCV ============
# acc: [0.66666667 1.         0.66666667 0.66666667 1.         0.66666667
#  0.66666667 1.         1.         1.        ]
#  평균 acc: 0.8333
# cross_val_precit acc: 0.8583333333333333
# ============ SGDClassifier ============
# acc: [0.66666667 0.66666667 0.66666667 0.66666667 1.         0.66666667
#  0.66666667 0.66666667 1.         0.66666667]
#  평균 acc: 0.7333
# cross_val_precit acc: 0.8583333333333333
# ============ SVC ============
# acc: [1.         1.         1.         1.         1.         0.66666667
#  0.66666667 0.66666667 1.         1.        ]
#  평균 acc: 0.9
# cross_val_precit acc: 0.9666666666666667
