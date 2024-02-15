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
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
# print(x.shape)    # (581012, 54)
scaler = StandardScaler()
x = scaler.fit_transform(x)
pca = PCA(n_components=48)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0, stratify=y)


# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = ohe.fit_transform(y_test.reshape(-1,1)).toarray()

#2
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


# DecisionTreeClassifier model.score 0.9031264253074361
# DecisionTreeClassifier : [0.07997877 0.0251775  0.01073969 0.01218087 0.12847307 0.00787725
#  0.04031029 0.01339028 0.06071461 0.01102486 0.04076288 0.01054879
#  0.02109731 0.023898   0.00966451 0.02021607 0.01392723 0.02429141
#  0.02244878 0.03317969 0.01943313 0.0153104  0.01952057 0.01801832
#  0.01175446 0.00930476 0.00796923 0.03942164 0.00618317 0.01616576
#  0.00937497 0.00690637 0.01103949 0.01034959 0.00548316 0.00752532
#  0.01777615 0.02657472 0.00542392 0.00461799 0.02321774 0.01083301
#  0.01185274 0.00956844 0.03236005 0.00331104 0.00870006 0.02210195]

# 선택된 특성 수: 40
# 컬럼 줄인 DecisionTreeClassifier 의 정확도: 0.9047184668209943


# RandomForestClassifier model.score 0.9396315069318348
# RandomForestClassifier : [0.03296925 0.02699773 0.01420767 0.01424879 0.07048755 0.01114129
#  0.02887403 0.01667828 0.03955062 0.014315   0.02664633 0.01625098
#  0.02247515 0.01800065 0.01954797 0.02665052 0.01881324 0.02685683
#  0.02264495 0.0253141  0.01991164 0.02420719 0.03082674 0.01913318
#  0.01923428 0.0182692  0.01641388 0.02213314 0.01151248 0.01867365
#  0.0153832  0.01462569 0.01440863 0.03314777 0.01387607 0.01658975
#  0.01790707 0.01898857 0.01548156 0.00998875 0.03191983 0.01286523
#  0.0141857  0.01289077 0.02509134 0.00722955 0.01077324 0.02166097]

# 선택된 특성 수: 40
# 컬럼 줄인 RandomForestClassifier 의 정확도: 0.9419206044594374


# 에러: y should be a 1d array, got an array of shape (464809, 7) instead.
# XGBClassifier model.score 0.8217860124093181
# XGBClassifier : [0.04129658 0.02303069 0.01220314 0.00991397 0.05213839 0.00792712
#  0.01237868 0.02009758 0.01842578 0.01398199 0.02090597 0.01029363
#  0.01692172 0.0160676  0.01294554 0.03466486 0.01374685 0.01995742
#  0.013517   0.01314523 0.01262384 0.01118793 0.05155484 0.01758722
#  0.02190599 0.02334453 0.02162252 0.0306061  0.01155879 0.01425376
#  0.00786328 0.00994025 0.01419333 0.18886624 0.01738632 0.0099564
#  0.00935178 0.03898581 0.01566164 0.00901974 0.01197776 0.00686931
#  0.00841667 0.00811883 0.01191359 0.0041266  0.00565075 0.02189643]

# 선택된 특성 수: 40
# 컬럼 줄인 XGBClassifier 의 정확도: 0.8210459282462587