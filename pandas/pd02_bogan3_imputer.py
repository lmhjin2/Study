import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6 ,8 ,10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]]).astype('float64')

data = data.transpose()
data.columns = ["x1", "x2", "x3", "x4"]
# print(data)
    #      x1   x2    x3   x4
    # 0   2.0  2.0   2.0  NaN
    # 1   NaN  4.0   4.0  4.0
    # 2   6.0  NaN   6.0  NaN
    # 3   8.0  8.0   8.0  8.0
    # 4  10.0  NaN  10.0  NaN

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

imputer = SimpleImputer()
data2 = imputer.fit_transform(data)
# print(data2)

imputer = SimpleImputer(strategy='mean')
data3 = imputer.fit_transform(data) # 평균
# print(data3)

imputer = SimpleImputer(strategy='median')
data4 = imputer.fit_transform(data) # 중위
# print(data4)

imputer = SimpleImputer(strategy='most_frequent')
data5 = imputer.fit_transform(data) # 가장 흔한놈
# print(data5)

imputer = SimpleImputer(strategy='constant')
data6 = imputer.fit_transform(data) # 상수
# print(data6)

imputer = SimpleImputer(strategy='constant', fill_value=777)
data7 = imputer.fit_transform(data) # 777넣기
# print(data7)

imputer = KNNImputer()  # KNN 알고리즘 으로 결측치 처리.
data8 = imputer.fit_transform(data)
# print(data8)

imputer = IterativeImputer()    
data9 = imputer.fit_transform(data)
print(data9)

print(np.__version__)   # 1.26.3 에서 mice 오류
                        # 1.22.4 에서 mice 정상
# from impyute.imputation.cs import mice    # predict 방식

# np.random.seed(3)

# aaa = mice(data.values,
#            n=10,    # 몇번 돌거냐.
#            random_seed=777 ) # seed, random_seed 안먹음. np.random.seed()로 고정해야됨
# print(aaa)
    # [[ 2.          2.          2.          1.99988929]
    #  [ 4.00022142  4.          4.          4.        ]
    #  [ 6.          5.99969225  6.          5.99986051]
    #  [ 8.          8.          8.          8.        ]
    #  [10.          9.99876899 10.          9.99962655]]
    # [[ 2.  2.  2.  2.]
    #  [ 4.  4.  4.  4.]
    #  [ 6.  6.  6.  6.]
    #  [ 8.  8.  8.  8.]
    #  [10. 10. 10. 10.]]