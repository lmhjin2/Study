import torch
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# 1. 데이터 로드 및 경로 설정
file_path = 'C:/Study/project/test/lotto_EN.csv'
df = pd.read_csv(file_path)

# 2. 컬럼명 매핑 (실제 파일 컬럼명 확인 결과에 따름)
round_col = 'drwNo'
no_cols = ['drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6']

# 1206회차를 예측하기 위해 1205회차 이하 데이터만 사용
train_df = df[df[round_col] < 1206].copy()
target_round = 1206

def prepare_data(data):
    X = []
    y = []
    # 데이터가 내림차순(최신순)이므로 오름차순(과거순)으로 정렬하여 시계열 학습
    data_sorted = data.sort_values(by=round_col).reset_index(drop=True)
    
    for i in range(len(data_sorted) - 1):
        # i회차 번호를 입력(X)으로, i+1회차 번호를 정답(y)으로 설정
        # .astype(int)를 추가하여 'No matching signature' 에러 방지
        features = data_sorted.iloc[i][no_cols].values.astype(int)
        labels = data_sorted.iloc[i+1][no_cols].values.astype(int)
        X.append(features)
        y.append(labels)
        
    return np.array(X), np.array(y)

X_train_raw, y_train_raw = prepare_data(train_df)

# 3. 모델 학습 및 예측
predicted_numbers = []

print(f"학습 시작 (데이터 크기: {len(X_train_raw)}개 회차)...")

for i in range(6):
    # 각 번호 위치별(1~6번째 숫자)로 모델 생성
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        verbose=1,
        # GPU가 있다면 사용, 없으면 CPU 사용
        task_type="GPU" if torch.cuda.is_available() else "CPU"
    )
    
    # y_train_raw[:, i]를 명시적으로 int 타입으로 변환하여 학습
    model.fit(X_train_raw, y_train_raw[:, i].astype(int))
    
    # 가장 마지막 회차(1205회) 데이터를 넣어 1206회 예측
    last_session_data = train_df.sort_values(by=round_col).iloc[-1][no_cols].values.astype(int)
    last_features = last_session_data.reshape(1, -1)
    
    pred = model.predict(last_features)
    predicted_numbers.append(int(pred[0][0]))

# 4. 결과 정리 (중복 제거 및 6개 번호 맞추기)
final_prediction = sorted(list(set(predicted_numbers)))

# 중복으로 인해 6개가 안 될 경우 랜덤으로 보충
while len(final_prediction) < 6:
    extra = np.random.randint(1, 46)
    if extra not in final_prediction:
        final_prediction.append(extra)

final_prediction.sort()

print(f"\n--- {target_round}회차 CatBoost 예측 결과 ---")
print(f"추천 번호: {final_prediction}")

# 5. 실제 1206회 당첨 번호와 비교
actual_row = df[df[round_col] == 1206]
if not actual_row.empty:
    actual_1206 = sorted(actual_row[no_cols].values[0].tolist())
    print(f"실제 번호: {actual_1206}")
    
    # 맞춘 개수 확인
    matched = set(final_prediction) & set(actual_1206)
    print(f"적중 번호: {sorted(list(matched))} (총 {len(matched)}개 적중)")
else:
    print("실제 1206회 데이터가 파일에 없습니다.")