import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron, LogisticRegression
                                            # regression이지만 의외로 분류
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#1
datasets = load_iris()
x = datasets.data
y = datasets['target']

### DataFrame
df = pd.DataFrame(x, columns = datasets.feature_names)
df['Target(Y)'] = y
# print(df)
print("================== 상관계수 히트맵 ======================================================")
print(df.corr())

# ================== 상관계수 히트맵 ======================================================
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target(Y)
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941   0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126  -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865   0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000   0.956547
# Target(Y)                   0.782561         -0.426658           0.949035          0.956547   1.000000

import matplotlib.pyplot as plt
import seaborn as sns
print(sns.__version__) # 0.12.2
## matplotlib 3.8.0 에선 안먹힘. 3.7.2로 다운
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), 
            square=True, 
            annot=True,     # 표 안에 수치 명시 # base에서만 나오네 뭐냐
            cbar=True       # 사이드 바
            )       
plt.show()