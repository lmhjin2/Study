# https://dacon.io/competitions/open/235576/mysubmission
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error, mean_absolute_error
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1 
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.dropna()  # 결측치 드랍.
test_csv = test_csv.fillna(test_csv.mean()) # 결측치에 평균치넣기
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state= 0)

n_splits =  10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = StandardScaler()

scaler.fit(x)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2
model = HistGradientBoostingRegressor()


#3
scores = cross_val_score(model, x, y, cv = kfold)
print('r2:', scores, "\n 평균 r2:", round(np.mean(scores), 4))
y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

r2 = r2_score(y_test, y_predict)
print('cross_val_precit r2:', r2)
# #4
# y_submit = model.predict(test_csv)

# submission_csv['count']=y_submit
# submission_csv.to_csv(path+"submission_0131.csv",index=False)


# HistGradientBoostingRegressor 의 정답률: 0.8275995871226449

# acc: [0.72827916 0.83447145 0.76077577 0.79893169 0.79301325 0.71022838
#  0.82958819 0.8153608  0.81599743 0.72366471]
#  평균 acc: 0.781

# acc: [0.80625957 0.79718249 0.8102575  0.84306899 0.78454678 0.75124243
#  0.75766769 0.81802511 0.74310886 0.69221838]
#  평균 acc: 0.7804


# r2: [0.80625957 0.7960218  0.8102575  0.84306899 0.78430097 0.75124243
#  0.75766769 0.81802511 0.74307291 0.69221838]
#  평균 r2: 0.7802
