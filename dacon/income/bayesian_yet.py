import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV

#1
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

train = pd.read_csv('d:/data/income/train.csv')
test = pd.read_csv('d:/data/income/test.csv')

labels = train.columns.tolist()
# print(labels)
# ['ID', 'Age', 'Gender', 'Education_Status', 'Employment_Status', 
# 'Working_Week (Yearly)', 'Industry_Status', 'Occupation_Status', 'Race', 'Hispanic_Origin', 
# 'Martial_Status', 'Household_Status', 'Household_Summary', 'Citizenship', 'Birth_Country', 
# 'Birth_Country (Father)', 'Birth_Country (Mother)', 'Tax_Status', 'Gains', 'Losses', 
# 'Dividends', 'Income_Status', 'Income']

# print(pd.value_counts(train['Tax_Status']))
# print(np.unique(train['Race']))

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

n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 42 )

parameters = [{'seed': 42,}]

#2
model = XGBRegressor() 
model.fit(x_train, y_train) 
model.score(x_test, y_test)
preds = model.predict(test_x)

submission = pd.read_csv('d:/data/income/sample_submission.csv')
submission['Income'] = preds
# print(submission)

submission.to_csv('c:/Study/dacon/income/output/0313_1.csv', index=False)

