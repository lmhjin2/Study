from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import sklearn.preprocessing
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler

path = "C:\\_data\\DACON\\와인품질분류\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submit_csv = pd.read_csv(path+"sample_submission.csv")

# print(train_csv.isna().sum(),test_csv.isna().sum()) 결측치 존재안함

# print(train_csv,test_csv,sep='\n') #[5497 rows x 13 columns], [1000 rows x 12 columns]
x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']

print(np.unique(y,return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)
print(np.unique(y,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))

x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
# print(x)
test_csv.loc[test_csv['type'] == 'red', 'type'] = 1 
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=42,stratify=y)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

############## smote ############## 
print("===== smote =====")
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=47, k_neighbors=3)
x_train, y_train = smote.fit_resample(x_train,y_train)
print(pd.value_counts(y_train))

model = RandomForestClassifier()

# fit & pred
model.fit(x_train,y_train,)
# model.fit(x_train,y_train)
pred = model.predict(x_test)
result = model.score(x_test,y_test)

print(result)

from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_test,pred)
f1  = f1_score(y_test,pred,average='macro')
print("ACC: ",acc)
print("F1 : ",f1)

# 기본
# ACC:  0.6518181818181819
# F1 :  0.38253859683408253

# SMOTE
# ACC:  0.6281818181818182
# F1 :  0.3760407800734257