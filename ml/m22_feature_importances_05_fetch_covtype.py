import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold,\
    cross_val_predict, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
import time as tm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import all_estimators
import time as tm
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

ohe = OneHotEncoder()
y_ohe = ohe.fit_transform(y.reshape(-1,1)).toarray()
x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = RobustScaler()

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2
models = [DecisionTreeClassifier(random_state= 0), RandomForestClassifier(random_state= 0),
          GradientBoostingClassifier(random_state= 0), XGBClassifier(random_state= 0)]

# np.set_printoptions(suppress=True)

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "accuracy score", accuracy_score(y_predict, y_test))
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_, end='\n\n')
        ## type(model).__name__ == 모델 이름만 뽑기
        # end = '\n\n' == print('\n') 한줄 추가
    except Exception as e:
        print("에러:", e)
        continue

# 최적의 매개변수 :  RandomForestClassifier(n_jobs=2)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 2}
# best_score : 0.9531721667795265
# model.score : 0.9553539925819471
# accuracy_score: 0.9553539925819471
# 최적 튠 ACC: 0.9553539925819471
# 걸린시간: 21265.71초


# DecisionTreeClassifier accuracy score 0.9384009018700034
# DecisionTreeClassifier model.score 0.9384009018700034
# DecisionTreeClassifier : [3.38586957e-01 2.62063436e-02 1.62351522e-02 6.15760889e-02
#  4.43984569e-02 1.50986192e-01 2.77596478e-02 3.36411471e-02
#  2.41185038e-02 1.41948547e-01 1.09236712e-02 2.98080268e-03
#  1.28575995e-02 2.10453126e-03 1.40039820e-04 1.00629533e-02
#  2.17944718e-03 1.17372426e-02 4.59968756e-04 6.44463341e-04
#  0.00000000e+00 1.21125986e-04 1.19422356e-04 2.90132415e-03
#  1.80410133e-03 6.96640730e-04 3.03195085e-03 7.47916858e-05
#  0.00000000e+00 1.14023352e-03 1.33755752e-03 0.00000000e+00
#  9.88098089e-04 3.10367922e-03 5.58221344e-04 8.62027520e-03
#  9.06161191e-03 4.90417149e-03 5.26959663e-05 4.03392764e-04
#  8.56031033e-04 1.18790676e-04 7.36850123e-03 2.52294451e-03
#  4.16629999e-03 1.32336160e-02 4.85541152e-03 3.17038059e-04
#  1.10026841e-03 2.00507414e-05 2.61508228e-04 2.25862027e-03
#  3.61190416e-03 8.41964596e-04]

# RandomForestClassifier accuracy score 0.9484694887395334
# RandomForestClassifier model.score 0.9484694887395334
# RandomForestClassifier : [2.43081178e-01 4.73712321e-02 3.24991090e-02 6.08751901e-02
#  5.75801092e-02 1.17906792e-01 4.10853248e-02 4.34574834e-02
#  4.12258208e-02 1.11854549e-01 1.11837182e-02 5.03658380e-03
#  1.17622631e-02 3.31727605e-02 1.15078721e-03 9.46295423e-03
#  2.35561745e-03 1.21406749e-02 4.72574377e-04 2.58941198e-03
#  9.25942175e-06 4.10434210e-05 1.28947073e-04 1.07823681e-02
#  2.79083331e-03 9.59566615e-03 4.16919755e-03 3.78480003e-04
#  5.64062362e-06 8.00163662e-04 1.73210788e-03 1.98103589e-04
#  1.00002727e-03 1.85331563e-03 7.12491237e-04 1.47820756e-02
#  9.85773869e-03 3.82246013e-03 1.63890544e-04 4.63152428e-04
#  6.50314292e-04 1.76171398e-04 5.44840700e-03 3.41322838e-03
#  3.90302602e-03 5.79970746e-03 4.55801241e-03 5.50311806e-04
#  1.53974041e-03 8.37094457e-05 5.42238605e-04 8.86504591e-03
#  9.55629018e-03 5.36270049e-03]

# 에러: y should be a 1d array, got an array of shape (464809, 7) instead.
# XGBClassifier accuracy score 0.8299441494625784
# XGBClassifier model.score 0.8299441494625784
# XGBClassifier : [0.0738029  0.00753619 0.00466608 0.01301255 0.00804899 0.01396225
#  0.0084062  0.01108591 0.00531119 0.01240367 0.05232818 0.026619
#  0.03143248 0.01832512 0.0034995  0.04015892 0.01655195 0.03779243
#  0.00595149 0.00568705 0.00175606 0.01253628 0.01275508 0.01316932
#  0.01170049 0.04387572 0.01228283 0.00427358 0.         0.00750463
#  0.01020971 0.00491347 0.01074402 0.01425536 0.02405372 0.0661288
#  0.03379748 0.02230686 0.01103471 0.00883834 0.02215048 0.00420664
#  0.02572436 0.01830724 0.02898455 0.04216641 0.01723602 0.0086262
#  0.01897692 0.00418631 0.01241479 0.02386761 0.03821564 0.0122184 ]