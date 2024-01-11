import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (581012, 54)  (581012,)
print(pd.value_counts(y))

# 2    283301       # 1    211840       
# 1    211840       # 2    283301
# 3     35754       # 3     35754
# 7     20510       # 4      2747
# 6     17367       # 5      9493
# 5      9493       # 6     17367
# 4      2747       # 7     20510


