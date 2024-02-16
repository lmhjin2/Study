import pandas as pd

data = [
    ["삼성","1000","2000"],
    ["현대","1100","3000"],
    ["LG","2000","500"],
    ["아모레","3500","6000"],
    ["네이버","100","1500"],
]

index = ["031", "059", "033", "045","023"]
columns = ["종목명","시가","종가"]

df = pd.DataFrame(data, index=index, columns=columns)
# print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500
### ========================== 시가가 1100원 이상인 행을 모두 출력 =========================
    ### 1.
print(df.loc[df["시가"] >= "1100"])
    ### 2.
print(df[df['시가'] >= "1100"])
    ### 3.
print(df[df["시가"].astype(int) >= 1100])
### ========================== 시가가 1100원 이상인 종가만 출력 =========================
    ### 1.
print(df.loc[df["시가"] >= "1100"].iloc[:,2:])    ## 이거만 컬럼명 위에 나옴
    ### 2.
filtered = df[df["시가"] >= "1100"]                 ## Name: 종가, dtype: object
print(filtered["종가"])
    ### 3. 
print(df.loc[df["시가"] >= "1100", "종가"])         ## Name: 종가, dtype: object
    ### 4.
print(df[df["시가"].astype(int) >= 1100])

### ========================== 쌤꺼 =========================
## 1. 시가 1100이상
aaa = df['시가'] >= '1100'
print(aaa)  # boolian
    # 031    False
    # 059     True
    # 033     True
    # 045     True
    # 023    False
    # Name: 시가, dtype: bool
print(df[aaa])  # True인것만 출력됨
print(df.loc[aaa])  # 위랑 같음
# print(df.iloc[aaa])    # Error
print(df[df['시가'] >= '1100'])    # 제일많이쓰는 표현

## 2. 시가 1100 이상의 종가
print(df[df['시가'] >= '1100']['종가'])
print(df[df["시가"] >= '1100'][2])    # Error
print(df.loc[df['시가'] >= '1100']['종가'])
print(df.loc[df['시가'] >= '1100', '종가'])

