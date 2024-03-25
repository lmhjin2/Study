import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *

#1
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

train = pd.read_csv('d:/data/income/train.csv')
test = pd.read_csv('d:/data/income/test.csv')

train_x = train.drop(columns=['ID', 'Income'])
train_y = train['Income']
test_x = test.drop(columns=['ID'])

encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    
    # train과 test 데이터셋에서 해당 열의 모든 값을 문자열로 변환
    train_x[i] = train_x[i].astype(str)
    test_x[i] = test_x[i].astype(str)
    
    le.fit(train_x[i])
    train_x[i] = le.transform(train_x[i])
    
    # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
    for case in np.unique(test_x[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
    
    test_x[i] = le.transform(test_x[i])

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state= 42)

# print(x_train.shape, x_test.shape) / (16000, 21) (4000, 21)
# print(y_train.shape, y_test.shape) / (16000,) (4000,)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 42 )

# best_rmse :  587.0828680513683

#2
model = Sequential()
model.add(Dense(32, input_shape = (21,)))
model.add(Dense(1, activation='relu'))


#3
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')

optimizer = Adam(learning_rate=0.001)

model.compile(loss='mse', optimizer=optimizer)
model.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_test, y_test), callbacks=[es,rlr]) 

#4
model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# mse, rmse
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)

print('RMSE : ', rmse)

# submission
preds = model.predict(test_x)

submission = pd.read_csv('d:/data/income/sample_submission.csv')
submission['Income'] = preds

submission.to_csv('c:/Study/dacon/income/output/best.csv', index=False)


