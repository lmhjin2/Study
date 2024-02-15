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

#1 데이터
x, y = load_digits(return_X_y=True)
print(x.shape, y.shape)     # 64 columns
scaler = StandardScaler()
x = scaler.fit_transform(x)
pca = PCA(n_components=50)
x = pca.fit_transform(x)

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

        # 남길 상위 특성 선택
        num_features_to_keep = 40
        sorted_indices = np.argsort(model.feature_importances_)[::-1]
        selected_features = sorted_indices[:num_features_to_keep]

        # 선택된 특성 수 출력
        print("선택된 특성 수:", len(selected_features))

        # 상위컬럼 데이터로 변환
        x_train_selected = x_train[:, selected_features]
        x_test_selected = x_test[:, selected_features]

        # 재학습, 평가
        model_selected = model.__class__(random_state=0)
        model_selected.fit(x_train_selected, y_train)
        y_predict_selected = model_selected.predict(x_test_selected)
        accuracy_selected = accuracy_score(y_test, y_predict_selected)

        # 프린트
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

# DecisionTreeClassifier model.score 0.8138888888888889
# DecisionTreeClassifier : [0.13228569 0.16419683 0.1332645  0.0974967  0.03979529 0.00653378
#  0.07463266 0.03828366 0.0035539  0.02785061 0.00784229 0.05397221
#  0.01717434 0.08207443 0.00456057 0.00077323 0.00692017 0.00834408
#  0.01016692 0.         0.00596957 0.         0.00305114 0.00802865
#  0.00454407 0.00457614 0.00462099 0.00309294 0.00147546 0.00441848
#  0.00783703 0.         0.         0.00424706 0.00189794 0.00152499
#  0.00077323 0.         0.00417546 0.00077323 0.00733256 0.00209878
#  0.00103098 0.00289065 0.00230129 0.00369434 0.00153358 0.00309294
#  0.00180421 0.00349244]

# 선택된 특성 수: 40
# 컬럼 줄인 DecisionTreeClassifier 의 정확도: 0.8138888888888889


# RandomForestClassifier model.score 0.9694444444444444
# RandomForestClassifier : [0.10715159 0.08816012 0.09552752 0.05689921 0.0370077  0.02149721
#  0.06082235 0.04555706 0.02761328 0.02926595 0.01403915 0.02906385
#  0.02322209 0.04340758 0.01601373 0.01433952 0.01407062 0.01494759
#  0.01933536 0.01117178 0.01257584 0.01063091 0.01206793 0.01244793
#  0.00848755 0.00728637 0.00759354 0.00827545 0.01186851 0.00738091
#  0.00916243 0.00744218 0.00645297 0.00832734 0.00613924 0.00750222
#  0.00705998 0.00536224 0.00929042 0.00569575 0.00897262 0.00550358
#  0.00611203 0.007111   0.0057037  0.00489853 0.00594384 0.00471308
#  0.0060721  0.00480656]

# 선택된 특성 수: 40
# 컬럼 줄인 RandomForestClassifier 의 정확도: 0.9583333333333334


# GradientBoostingClassifier model.score 0.925
# GradientBoostingClassifier : [0.15907677 0.15233242 0.15533384 0.14362159 0.01228672 0.00504414
#  0.09928383 0.04677825 0.01895612 0.01619664 0.00951405 0.0168416
#  0.01667904 0.05729355 0.00618558 0.0115681  0.00281088 0.01315259
#  0.00747271 0.0009069  0.00123642 0.000816   0.00098899 0.00262513
#  0.00415999 0.00152912 0.00099706 0.00108852 0.00877528 0.00090357
#  0.00070539 0.00035832 0.00052288 0.00072814 0.00277421 0.00147536
#  0.00026683 0.00074733 0.00241732 0.00127793 0.00215262 0.0001365
#  0.00031559 0.00264394 0.00427176 0.00126997 0.00144401 0.00080555
#  0.0011615  0.00006942]

# 선택된 특성 수: 40
# 컬럼 줄인 GradientBoostingClassifier 의 정확도: 0.9222222222222223


# XGBClassifier model.score 0.9194444444444444
# XGBClassifier : [0.08496775 0.0875864  0.07095738 0.07657386 0.0207785  0.01062588
#  0.06634784 0.03125646 0.04489446 0.01262553 0.02095792 0.02460307
#  0.0245776  0.06279167 0.01589054 0.04204896 0.01078644 0.01647599
#  0.00809671 0.01291663 0.00712776 0.01296361 0.00535286 0.01767939
#  0.01779621 0.00212894 0.01394713 0.0185055  0.02123099 0.00416882
#  0.00256611 0.0078784  0.00746962 0.00735328 0.00264117 0.00611597
#  0.00833214 0.00615688 0.01100844 0.01431817 0.00348673 0.00523016
#  0.00245127 0.00367296 0.01231724 0.00985417 0.00622399 0.00715821
#  0.00608806 0.00301615]

# 선택된 특성 수: 40
# 컬럼 줄인 XGBClassifier 의 정확도: 0.9222222222222223