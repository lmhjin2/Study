##############################################################################
"## joblib"
from joblib import dump, load

# 모델 저장
dump(model, 'xgboost_model.joblib')

# 모델 불러오기
loaded_model = load('xgboost_model.joblib')

##############################################################################
"## pickle"
import pickle

# 모델 저장
pickle.dump(model, open(path + 'm39_pickle1_save.dat', 'wb'))

# 모델 불러오기
model = pickle.load(open(path + 'm39_pickle1_save.dat', 'rb'))  # rb = read binary

##############################################################################
"## XGBoost 내장함수"
# 모델 저장
model.save_model('xgboost_model.xgb')

# 모델 불러오기
loaded_model = xgb.Booster(model_file='xgboost_model.xgb')
