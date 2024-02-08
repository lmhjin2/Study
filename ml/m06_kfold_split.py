import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold

#1 데이터
datasets = load_iris()

df = pd.DataFrame(datasets.data, columns = datasets.feature_names)
print(df) 

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

for train_index, val_index in kfold.split(df):
    print(train_index, "\n", val_index)
    print('훈련데이터 갯수:', len(train_index),
          '검증데이터 갯수', len(val_index))
