import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50]) # (13,)
# print(aaa.shape)   # (13,)
aaa = aaa.reshape(-1,1)    # (13, 1)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=0.1)  # 전체 데이터의 몇퍼센트를 이상치로 볼거냐

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
