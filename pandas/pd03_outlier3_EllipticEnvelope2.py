import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T # (13,2)
# aaa = aaa.reshape(-1,1)    # (26, 1)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=0.1)  # 전체 데이터의 몇퍼센트를 이상치로 볼거냐

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)

# aaa = aaa.reshape(-1,1)   # (26, 1) 안하면 숫자 13개만 나옴 + 이렇게 하면 이상함 == 할거면 컬럼별로 하는게 낫다.(쌤 생각)
# [-10,2, 3, 4, 5, 6, 7, 8, 9,10,11,12,50,100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]
# [ 1  1  1  1  1  1  1  1  1  1  1  1  1 -1   1    1  1  -1   1    -1    1   1    1   1   1   1 ]
# 100, 500, -70000