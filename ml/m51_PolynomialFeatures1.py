import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)
# print(x) #-> x1^2, x1*x2, x2^2 를 한다면 아래값
# [[0 1]    ->  0,  0,  1   -> y값
#  [2 3]    ->  4,  6,  9   -> y값
#  [4 5]    -> 16, 20, 25   -> y값
#  [6 7]]   -> 36, 42, 49   -> y값

pf = PolynomialFeatures(degree=2, include_bias=False) # degree=2 일때 그래프 모양 == U, ( 눕힌거
x_pf = pf.fit_transform(x)                            # degree=3 일때 그래프 모양 == ~ , s 눕힌거
# print(x_pf)    # x1, x2, x1^2, x1*x2, x2^2
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

pf2 = PolynomialFeatures(degree=3, include_bias=False)
x_pf2 = pf2.fit_transform(x)
# print(x_pf)   # x_pf, x1^3, x1^2*x2^1, x1^1*x2^2, x2^3
# [[  0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  6.   7.  36.  42.  49. 216. 252. 294. 343.]]


pf3 = PolynomialFeatures(degree=2, include_bias=True) # True가 기본값
x_pf3 = pf3.fit_transform(x)
# print(x_pf3)  # 0승 = 1, 모두 1을 포함하게됨. 전부 1이라 의미가 없는 데이터만 생성
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]

# ==================== 컬럼3개 ====================
print("="*20, "컬럼3개", "="*20)

x = np.arange(12).reshape(4,3)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)

# [[  0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  9.  10.  11.  81.  90.  99. 100. 110. 121.]]