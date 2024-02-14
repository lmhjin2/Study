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

#1 데이터
x, y = load_digits(return_X_y=True)
print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)

#2 모델
models = [DecisionTreeClassifier(random_state= 0), RandomForestClassifier(random_state= 0),
          GradientBoostingClassifier(random_state= 0), XGBClassifier(random_state= 0)]

np.set_printoptions(suppress=True)

for model in models:
    try:
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        print(type(model).__name__, "model.score", results)
        print(type(model).__name__, ":", model.feature_importances_, end='\n\n')
        ## type(model).__name__ == 모델 이름만 뽑기
        # end = '\n\n' == print('\n') 한줄 추가
        # 상위 특성 선택
        num_features_to_keep = 50
        sorted_indices = np.argsort(model.feature_importances_)[::-1]
        selected_features = sorted_indices[:num_features_to_keep]
        # 상위컬럼 데이터로 변환
        x_train_selected = x_train[:,selected_features]
        x_test_selected = x_test[:,selected_features]
        # 재학습, 평가
        model_selected = model.__class__(random_state=0)
        model_selected.fit(x_train_selected, y_train)
        y_predict_selected = model.predict(x_test_selected)
        accuracy_selected = accuracy_score(y_test, y_predict_selected)
        # 프린트
        print("선택된 특성 수:", num_features_to_keep)
        print("컬럼 줄인", type(model).__name__,"의 정확도:", accuracy_selected)
        print('\n')
    except Exception as e:
        print("에러:", e)
        continue

# import pandas as pd
# print(pd.DataFrame(model.cv_results_).transpose()) # 잘 안보이니까 dataframe에 담아서 따로 열던가 csv파일로 만들어서 보던가


# 최적의 매개변수 :  SVC(C=100, gamma=0.001)
# 최적의 파라미터 :  {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
# best_score : 0.9444444444444444
# model.score : 1.0
# accuracy_score: 1.0
# 최적 튠 ACC: 1.0
# 걸린시간: 0.16 초

# import sklearn as sk
# print(sk.__version__) # 1.1.3 근데 아직도 experimental임. 한국어로 실험실 옵션이라 생각하면 될듯?

# DecisionTreeClassifier accuracy score 0.8777777777777778
# DecisionTreeClassifier model.score 0.8777777777777778
# DecisionTreeClassifier : [0.         0.         0.00154647 0.01481623 0.00817679 0.05949888
#  0.         0.         0.         0.01019572 0.02059994 0.
#  0.0090116  0.00077323 0.00142751 0.         0.00153241 0.00511524
#  0.01043687 0.03032885 0.04280637 0.08949749 0.         0.
#  0.         0.         0.06258673 0.0439637  0.04879706 0.01522299
#  0.00266228 0.         0.         0.05987218 0.01097742 0.00077323
#  0.07819364 0.0223194  0.00993957 0.         0.         0.00407654
#  0.07754247 0.05548499 0.00796742 0.00427401 0.00939756 0.
#  0.         0.00220924 0.01030344 0.00577672 0.00192002 0.01011408
#  0.03405664 0.0026171  0.         0.00321599 0.00680268 0.00394349
#  0.06315035 0.02950627 0.00077323 0.00579594]

# RandomForestClassifier accuracy score 0.9805555555555555
# RandomForestClassifier model.score 0.9805555555555555
# RandomForestClassifier : [0.00000000e+00 1.73012280e-03 2.06956243e-02 8.67015483e-03
#  8.98573397e-03 2.21967581e-02 9.46780673e-03 9.44761131e-04
#  7.00384495e-05 1.12406044e-02 2.45436180e-02 6.69654146e-03
#  1.44409664e-02 2.73517514e-02 6.16442158e-03 6.40686882e-04
#  7.08705742e-05 7.74160767e-03 2.04393742e-02 2.69993282e-02
#  2.65815734e-02 4.84233179e-02 1.05815399e-02 4.26793240e-04
#  0.00000000e+00 1.44323263e-02 4.51747156e-02 2.33409695e-02
#  2.51175508e-02 2.20712484e-02 2.96608061e-02 4.03009941e-05
#  0.00000000e+00 3.09546756e-02 3.17977061e-02 2.15061240e-02
#  4.24864299e-02 2.19144624e-02 2.59033397e-02 0.00000000e+00
#  1.37194186e-05 9.18617846e-03 4.12743217e-02 4.30018892e-02
#  2.09601991e-02 1.59346717e-02 1.91720304e-02 1.12886335e-04
#  7.04457244e-05 2.40777734e-03 1.79310421e-02 2.51480545e-02
#  1.44041762e-02 2.07915234e-02 2.52653306e-02 1.71488138e-03
#  3.66823064e-05 1.77626245e-03 1.99159781e-02 1.09829294e-02
#  2.35991617e-02 2.64839908e-02 1.66737014e-02 3.63751527e-03]

# GradientBoostingClassifier accuracy score 0.9583333333333334
# GradientBoostingClassifier model.score 0.9583333333333334
# GradientBoostingClassifier : [0.00000000e+00 1.13546833e-03 5.93338712e-03 4.64370537e-03
#  1.71609099e-03 5.54942381e-02 1.06757880e-02 2.23184387e-04
#  2.23398884e-04 6.74591385e-03 1.91498224e-02 5.81414177e-04
#  5.32764759e-03 1.51630445e-02 3.70346536e-03 2.93344264e-04
#  3.90289472e-05 1.72488422e-03 1.28209038e-02 4.14162899e-02
#  1.74931636e-02 8.04920514e-02 6.67962775e-03 1.65238306e-07
#  0.00000000e+00 2.60293630e-03 5.48229032e-02 1.06659391e-02
#  3.52724694e-02 2.30713906e-02 7.75158960e-03 4.33346302e-04
#  0.00000000e+00 6.28494303e-02 5.27870091e-03 8.03473405e-03
#  7.02819340e-02 1.02061933e-02 1.94175885e-02 0.00000000e+00
#  0.00000000e+00 7.59048604e-03 8.40921710e-02 6.94504149e-02
#  9.56442079e-03 2.19252617e-02 2.15053620e-02 5.58917242e-04
#  1.19545481e-10 3.51227640e-04 3.41945140e-03 1.98267066e-02
#  1.27818209e-02 1.06844022e-02 2.67351902e-02 1.88927758e-04
#  7.68919750e-04 2.24327603e-04 1.52618908e-02 7.37156000e-04
#  5.67660690e-02 7.14331660e-03 2.54340292e-02 2.62434676e-03]

# XGBClassifier accuracy score 0.9611111111111111
# XGBClassifier model.score 0.9611111111111111
# XGBClassifier : [0.         0.03547805 0.01250185 0.01475842 0.0040258  0.03839405
#  0.0060233  0.02589995 0.         0.0103426  0.00881972 0.00881762
#  0.00779756 0.0114924  0.002698   0.00403895 0.         0.00504206
#  0.00794138 0.04761722 0.01065672 0.04359426 0.01169978 0.
#  0.         0.00645921 0.03122442 0.00896103 0.0255209  0.01867688
#  0.01143218 0.         0.         0.07488101 0.00505739 0.00405002
#  0.05529634 0.0080259  0.04107445 0.         0.         0.01319927
#  0.03374613 0.03835197 0.01176671 0.01554845 0.03349986 0.
#  0.         0.00340624 0.00633322 0.01644418 0.01096438 0.0107435
#  0.03194077 0.         0.         0.00354939 0.0120876  0.00538477
#  0.07396317 0.01670739 0.0376563  0.01640736]