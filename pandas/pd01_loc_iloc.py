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

datasets = pd.DataFrame(data, index=index, columns=columns)
print(datasets)
print("=====================================")
# datasets[0] <= 에러
# datasets["031"] <= 에러
print(datasets['종목명'])   # pandas에서는 column이 기준
# 031     삼성
# 059     현대
# 033     LG
# 045    아모레
# 023    네이버
# Name: 종목명, dtype: object
print(datasets['종목명']['045'])   # pandas에서는 column이 기준
# 아모레
print("=====================================")
'''
loc:  인덱스를 기준으로 행 데이터 추출
iloc: 행 번호를 기준으로 행 데이터 추출
'''
print(datasets.loc["023"])
print(datasets.iloc[4])
print(datasets.iloc[-1])

print("=====================================")
print(datasets['시가']['045'])      # 3500
print(datasets.loc['045']['시가'])  # 3500
print(datasets.loc['045'][1])       # 3500, 권장되지 않음
print(datasets.iloc[3][1])          # 3500, 권장되지 않음
print(datasets.loc['045','시가'])   # 3500
print(datasets.iloc[3,1])           # 3500
print(datasets.iloc[3].loc['시가']) # 3500
print(datasets.시가['045'])         # 3500

print("=====================================")
print(datasets.iloc[3:5,1])                 # 3500  100
print(datasets.iloc[[3,4],1])               # 3500  100
print(datasets.iloc[[3,4]]['시가'])         # 3500  100
print(datasets.loc[['045','023'],'시가'])   # 3500  100
print(datasets.loc['045':'023', '시가'])    # 3500  100 슬라이싱 시 뒤쪽도 포함을 하는것에 유의

for i in range(3):
    print(datasets.iloc[i]['시가'])
    print(datasets['시가'].iloc[i])
