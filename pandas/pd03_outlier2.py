import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T # (13,2)

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

# 이상치 위치 전부 뽑는 코드
for i in range(len(outliers_loc[0])):
    print("이상치 위치", outliers_loc[0][i], ",", outliers_loc[1][i])


#   quartile =   quantile    =   percentile
# 0 quartile = 0 quantile    = 0 percentile
# 1 quartile = 0.25 quantile = 25 percentile
# 2 quartile = 0.5 quantile  = 50 percentile (median)
# 3 quartile = 0.75 quantile = 75 percentile
# 4 quartile = 1 quantile    = 100 percentile


### 과제 // 이상치 결측치 해결해서 적용한 결과를 넣을것 // pandas 폴더 참조
# pd04_1_따릉이.py
# pd04_2_kaggle_bike.py
# pd04_3_대출.py
# pd04_4_캐글_비만.py
# 깃허브 잔디 채우기
