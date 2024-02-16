import pandas as pd
from datetime import datetime
import numpy as np

dates = ['2/16/2024', '2/17/2024', '2/18/2024', '2/19/2024', '2/20/2024', '2/21/2024']
dates = pd.to_datetime(dates)
print(dates)

print(f"=================================================")
ts = pd.Series([2,np.nan, np.nan,8,10,np.nan], index=dates)
print(ts)
    # 2024-02-16     2.0
    # 2024-02-17     NaN
    # 2024-02-18     NaN
    # 2024-02-19     8.0
    # 2024-02-20    10.0
    # 2024-02-21     NaN
print("=" * 50)

ts = ts.interpolate()   # 데이터 보간
print(ts)               # 알아서 중간값으로 맞춰주지만 최댓값 이상은 모르니 최댓값 까지만 출력됨
    # 2024-02-16     2.0
    # 2024-02-17     4.0
    # 2024-02-18     6.0
    # 2024-02-19     8.0
    # 2024-02-20    10.0
    # 2024-02-21    10.0
    # dtype: float64
print(f"{'=' * 50}")
    # print("=" * 20)
    # print("=" * 20)
"""
결측치 처리
1. 행 또는 열 삭제
2. 임의의 값
    평균 : mean
    중위 : median
    0 : fillna
    앞값: ffill (front-fill)
    뒷값: bfill (back-fill)
    특정값 : 123, 777, 등등 대신 뭔가 조건을 같이 넣는게 좋겠지?
    기타등등
3. 보간 : interpolate
4. 모델 학습 후 결측치 predict해서 넣기
    ex) x2 맞추기
    데이터 분리 후
    하고싶은대로 학습. predict
5. 부스팅 계열 : 통상 결측치, 이상치에 대해 자유롭다
    알고리즘상 결측치, 이상치를 자동 처리함
    



"""
