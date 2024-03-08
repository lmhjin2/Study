import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
import time as tm
import warnings
warnings.filterwarnings('ignore')


path = 'c:/_data/kaggle/Obesity_Risk/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder

lae_G = LabelEncoder()
train_csv['Gender'] = lae_G.fit_transform(train_csv['Gender'])
test_csv['Gender'] = lae_G.transform(test_csv['Gender'])

lae_fhwo = LabelEncoder()
train_csv['family_history_with_overweight'] = lae_fhwo.fit_transform(train_csv['family_history_with_overweight'])
test_csv['family_history_with_overweight'] = lae_fhwo.transform(test_csv['family_history_with_overweight'])

lae_FAVC = LabelEncoder()
train_csv['FAVC'] = lae_FAVC.fit_transform(train_csv['FAVC'])
test_csv['FAVC'] = lae_FAVC.transform(test_csv['FAVC'])

lae_CAEC = LabelEncoder()
train_csv['CAEC'] = lae_CAEC.fit_transform(train_csv['CAEC'])
test_csv['CAEC'] = lae_CAEC.transform(test_csv['CAEC'])

lae_SMOKE = LabelEncoder()
train_csv['SMOKE'] = lae_SMOKE.fit_transform(train_csv['SMOKE'])
test_csv['SMOKE'] = lae_SMOKE.transform(test_csv['SMOKE'])

lae_SCC = LabelEncoder()
train_csv['SCC'] = lae_SCC.fit_transform(train_csv['SCC'])
test_csv['SCC'] = lae_SCC.fit_transform(test_csv['SCC'])

lae_CALC = LabelEncoder()
test_csv['CALC'] = lae_CALC.fit_transform(test_csv['CALC'])
train_csv['CALC'] = lae_CALC.transform(train_csv['CALC'])

lae_MTRANS = LabelEncoder()
train_csv['MTRANS'] = lae_MTRANS.fit_transform(train_csv['MTRANS'])
test_csv['MTRANS'] = lae_MTRANS.transform(test_csv['MTRANS'])

lae_NObeyesdad = LabelEncoder()
train_csv['NObeyesdad'] = lae_NObeyesdad.fit_transform(train_csv['NObeyesdad'])

x = train_csv.drop(['NObeyesdad'], axis = 1)
y = train_csv['NObeyesdad']

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

# y = np.array(y.values.reshape(-1,1))
# y_ohe = OneHotEncoder(sparse=False).fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state= 5 )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

#2. model
start_t = tm.time()
bayesian_params = {
    'learning_rate' : (0.001, 1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : ( -0.001, 10),
    'reg_alpha' : (0.01, 50)
}
def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),      # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),    # min은 최소값을 반환, max는 최대값을 반환. 0과 1사이로 숫자를 뽑음
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),   # 10이상의 정수
        'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
        'reg_alpha' : reg_alpha,       
    }
    model = XGBClassifier(**params, n_jobs=-1)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train),(x_test, y_test)],
            #   eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50,
              )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    return results

bay = BayesianOptimization(
    f=xgb_hamsu,
    pbounds=bayesian_params,
    random_state=777
)

n_iter = 100
bay.maximize(init_points=5, n_iter=n_iter)

print(bay.max)
end_t = tm.time()

print(n_iter, "번 도는데 걸린시간 : ", np.round(end_t - start_t, 2), '초')

# https://www.kaggle.com/c/playground-series-s4e2/overview

# 점수 : 0.91221
# 최적의 파라미터 :  {'seed': 315}
# model.score : 0.9210019267822736
# 최적 튠 ACC: 0.9210019267822736

# 0개 컬럼이 삭제되었을 때 Score:  0.9087186897880539

# T acc:  0.9202793834296724
# F acc:  0.914980732177264

# VS acc:  0.9161849710982659
# VH acc:  0.9144990366088632

# model.score :  0.9142581888246628
# 스태킹 ACC : 0.9142581888246628

