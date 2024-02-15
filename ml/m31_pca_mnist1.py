from keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()   # y가 필요가없어서 _넣음. 안넣으면 x에 다들어감
# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
# x = np.append(x_train, x_test, axis=0)      # concate, append,vhstack 뭘쓰던 ㄱㅊ
# x = np.vstack([x_train, x_test])           # concate, append,vhstack 뭘쓰던 ㄱㅊ
x = np.concatenate([x_train, x_test], axis=0)      # concatenate, append,vhstack 뭘쓰던 ㄱㅊ
# print(x.shape)      # (70000, 28, 28)

############# [실습] ############ // x = (70000, 784)
# pca를 통해 0.95 이상인 n_componets는 몇개?
# 0.95 이상     // 332 // 631
# 0.99 이상     // 544 // 454
# 0.999 이상    // 683 // 299
# 1.0 일때 몇개?    // 1? 스케일러를 풀던가 소수점 아래 멀리서 반올림하던가
#                  713 // 72
# 힌트 : np.argmax
############### [ver.1] ################## // x = (70000, 784)
# scaler = StandardScaler()
# x = scaler.fit_transform(x.reshape(70000, 28*28))
# pca = PCA(n_components=784)
# pca.fit(x)
# print(x.shape)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# dim = np.argmax(np.round(cumsum,10) >= 1) + 1 
# print(dim) 
# # print(cumsum)
# ############### [ver.2] ##################
# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# pca = PCA(n_components=x.shape[1])
# x1 = pca.fit_transform(x)
# EVR = pca.explained_variance_ratio_
# EVR_sum = np.cumsum(EVR)    # == 윗 코드 cumsum
# # print(EVR_sum)
# evr_sum = pd.Series(EVR_sum)
# print(len(EVR_sum[EVR_sum >= 0.95]))    # 631 
# print(len(EVR_sum[EVR_sum >= 0.99]))    # 454
# print(len(EVR_sum[EVR_sum >= 0.999]))   # 299
# print(len(EVR_sum[EVR_sum >= 1.0]))     # 72

############### [ver.3] ##################
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
pca = PCA(n_components=784)
x = pca.fit_transform(x)
evr = pca.explained_variance_ratio_
cum_sum = np.cumsum(evr)
print(np.argmax(cum_sum >= 0.95)+1)
print(np.argmax(cum_sum >= 0.99)+1)
print(np.argmax(cum_sum >= 0.999)+1)
print(np.argmax(cum_sum >= 1.0)+1)


