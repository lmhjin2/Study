import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

train = pd.read_csv('d:/data/income/train.csv')
test = pd.read_csv('d:/data/income/test.csv')

print(train.head(3))
print(test.head(3))

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

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor() 
model.fit(train_x, train_y) 

preds = model.predict(test_x)

submission = pd.read_csv('./sample_submission.csv')
submission['Income'] = preds
print(submission)

submission.to_csv('c:/Study/dacon/income/output/0313_1.csv', index=False)