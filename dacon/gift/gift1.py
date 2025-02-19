import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist

# 데이터 파일 경로
data_path = './data.csv'  # 참가자가 제공받은 경로로 설정
data = pd.read_csv(data_path)

print("데이터 로드 완료!")
print(f"총 데이터 포인트 수: {len(data)}")
print(data.head())  # 데이터 확인

# 산타의 썰매 용량
santa_capacity = 25

# 출발점 설정 및 경로 초기화
route = ["DEPOT"]
total_distance = 0
current_capacity = santa_capacity

# 모든 포인트 간 거리 계산 (거리 행렬 생성)
points = data[['x', 'y']].values
distance_matrix = squareform(pdist(points, metric='euclidean'))
distance_df = pd.DataFrame(distance_matrix, index=data['point_id'], columns=data['point_id'])

# DEPOT과 방문해야 할 포인트 분리
remaining_points = data[data['point_id'] != 'DEPOT'].copy()
current_position = 'DEPOT'  # 출발점

print("데이터 전처리 및 초기 설정 완료!")

while not remaining_points.empty:
    # 현재 용량으로 방문 가능한 포인트 필터링
    feasible_points = remaining_points[remaining_points['demand'] <= current_capacity].copy()

    # 방문 가능한 포인트가 없는 경우: DEPOT으로 복귀
    if feasible_points.empty:
        # DEPOT 복귀
        total_distance += distance_df.loc[current_position, 'DEPOT']
        route.append("DEPOT")
        current_position = "DEPOT"
        current_capacity = santa_capacity
        continue

    # 가장 가까운 포인트 선택 (거리 계산 후 추가)
    feasible_points['distance'] = feasible_points['point_id'].apply(
        lambda x: distance_df.loc[current_position, x]
    )
    nearest_point = feasible_points.loc[feasible_points['distance'].idxmin()]

    # 경로 업데이트
    route.append(nearest_point['point_id'])
    total_distance += distance_df.loc[current_position, nearest_point['point_id']]
    current_position = nearest_point['point_id']
    current_capacity -= nearest_point['demand']

    # 방문한 포인트 제거
    remaining_points = remaining_points[remaining_points['point_id'] != nearest_point['point_id']].copy()

# 마지막으로 DEPOT으로 복귀
total_distance += distance_df.loc[current_position, 'DEPOT']
route.append("DEPOT")

print("탐욕 알고리즘 실행 완료!")
print(f"총 이동 거리: {total_distance}")

# 경로를 CSV 파일로 저장
output_file = './santa_route.csv'
route_df = pd.DataFrame(route, columns=['point_id'])
route_df.to_csv(output_file, index=False)

print(f"최종 경로 제출 Submission 파일 저장 완료: {output_file}")
# https://dacon.io/competitions/official/236437/mysubmission
