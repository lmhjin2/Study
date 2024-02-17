import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']
# print(x.shape)        # (10886, 8)
# print(train_csv.shape)    #(10886, 11)

def fit_outlier(data):  
    data = pd.DataFrame(data)
    for label in data:
        series = data[label]        # data의 label이라는 컬럼의 데이터를 series에 담음
        q1 = series.quantile(0.25)  # q1 = 25퍼센트 지점  
        q3 = series.quantile(0.75)  # q3 = 75퍼센트 지점
        iqr = q3 - q1
        upper_bound = q3 + (iqr * 1.5)     # 이상치 범위 설정
        lower_bound = q1 - (iqr * 1.5)
        
        series[series > upper_bound] = np.nan   # series안에 이상치들 전부 np.nan(결측치) 처리
        series[series < lower_bound] = np.nan
        print(series.isna().sum())      # series 안에 결측치 갯수
        series = series.interpolate()   # 결측치 interpolate()로 채우기
        data[label] = series    # 원래 위치에 덮어쓰기
        
    # data = data.fillna(data.ffill())
    # data = data.fillna(data.bfill())  
    return data
# print(x.isna().sum())
x = fit_outlier(x)
# print(x.isna().sum())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
np.set_printoptions(suppress=True)
# 2
model = XGBRegressor(random_state=0)
# 3
model.fit(x_train,y_train)

# 4 
results = model.score(x_test,y_test)
print("model.score",results)
y_predict = model.predict(x_test)
print('r2', r2_score(y_test, y_predict))

y_submit = model.predict(test_csv)

submission_csv['count']=y_submit
submission_csv.to_csv(path+"submission_0217.csv",index=False)

# results 0.3144480634376666
# r2 0.3144480634376666
