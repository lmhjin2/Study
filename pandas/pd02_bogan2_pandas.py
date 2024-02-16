import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6 ,8 ,10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

data = data.transpose()
data.columns = ["x1", "x2", "x3", "x4"]
# print(data)
    #      x1   x2    x3   x4
    # 0   2.0  2.0   2.0  NaN
    # 1   NaN  4.0   4.0  4.0
    # 2   6.0  NaN   6.0  NaN
    # 3   8.0  8.0   8.0  8.0
    # 4  10.0  NaN  10.0  NaN
# print(data.isnull())    # True가 결측치임
    #       x1     x2     x3     x4
    # 0  False  False  False   True
    # 1   True  False  False  False
    # 2  False   True  False   True
    # 3  False  False  False  False
    # 4  False   True  False   True
# print(data.isna().sum())
    # x1    1
    # x2    2
    # x3    0
    # x4    3
# print(data.info())
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 5 entries, 0 to 4
    # Data columns (total 4 columns):
    #  #   Column  Non-Null Count  Dtype  
    # ---  ------  --------------  -----  
    #  0   x1      4 non-null      float64
    #  1   x2      3 non-null      float64
    #  2   x3      5 non-null      float64
    #  3   x4      2 non-null      float64
    # dtypes: float64(4)
    # memory usage: 288.0 bytes
    # None
# print(data.dropna())
# print(data.dropna(axis=0))    # 기본값 axis = 0
# print(data.dropna(axis=1))

# 2-1 특정값 - 평균
# means = data.mean()
# print(means)
# data2 = data.fillna(means)
# print(data2)

# 2-2 특정값 - 중위
# median = data.median()
# print(median)
# data3 = data.fillna(median)
# print(data3)

# 2-3 특정값 - 임의의 값 0 채우기
# data4 = data.fillna(0)
# print(data4)

# data4 = data.fillna(777)    # 777 채우기
# print(data4)

# 2-4 특정값 - front-fill
# data5 = data.fillna(method='ffill')
# data5 = data.ffill()
# print(data5)

# 2-5 특정값 - back-fill
# data9 = data.bfill()
# print(data9)



######################### 특정 컬럼만 #################

means = data['x1'].mean()
print(means)    # 6.5
median = data['x4'].median()
print(median)   # 6.0

data['x1'] = data['x1'].fillna(means)
data['x4'] = data['x4'].fillna(median)
data['x2'] = data['x2'].ffill()
print(data)

