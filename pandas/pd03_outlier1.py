import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50]) # 13개

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 :", quartile_1)      # 4
    print("q2 : ", q2)                  # 7 ==median
    print("3사분위 :", quartile_3)      # 10
    iqr = quartile_3 - quartile_1       # (0% ~ 75%) - (0% ~ 25%) == (25% ~ 75%)
    print("iqr : ", iqr)                # 6
    lower_bound = quartile_1 - (iqr * 1.5)      # -5
    upper_bound = quartile_3 + (iqr * 1.5)      # 19
    return np.where((data_out>upper_bound) |    # shift + \ == |   // or 과 같은뜻. python 문법
                    (data_out<lower_bound))     # np.where 이면 위치 반환해줌
    # 19보다 크거나 -5보다 작은놈들의 위치를 찾는다

outliers_loc = outliers(aaa)
print("이상치의위치 :", outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

#   quartile =   quantile    =   percentile
# 0 quartile = 0 quantile    = 0 percentile
# 1 quartile = 0.25 quantile = 25 percentile
# 2 quartile = 0.5 quantile  = 50 percentile (median)
# 3 quartile = 0.75 quantile = 75 percentile
# 4 quartile = 1 quantile    = 100 percentile


