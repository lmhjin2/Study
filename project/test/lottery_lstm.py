import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터 로드 및 전처리
file_path = 'C:/Study/project/test/lotto_EN.csv' 
df = pd.read_csv(file_path)

# 과거순 정렬 및 필요한 컬럼 추출
round_col = 'drwNo'
no_cols = ['drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6']
df_sorted = df.sort_values(by=round_col).reset_index(drop=True)

# 1206회 예측을 위해 1205회까지만 사용
train_data = df_sorted[df_sorted[round_col] < 1206][no_cols].values

# 데이터 정규화 (0~1 사이 값으로 변환, 1~45 범위를 학습하기 좋게 만듦)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(train_data)

# 2. LSTM용 데이터셋 생성 (Window 설정)
# WINDOW_SIZE: 과거 몇 회차를 보고 다음을 예측할 것인가?
WINDOW_SIZE = 50 

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, WINDOW_SIZE)
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# 3. LSTM 모델 정의
class LottoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LottoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # 마지막 타임스텝의 결과만 사용
        return out

class LottoLSTM_Enhanced(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LottoLSTM_Enhanced, self).__init__()
        # 1. LSTM 층
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # 2. 중간 연결 층 (Dense Layer)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        
        # 3. 최종 출력 층
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :]) # 마지막 시점의 hidden state 추출
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 6   # 번호 6개
hidden_size = 128
num_layers = 2
output_size = 6  # 예측할 번호 6개

model = LottoLSTM(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 모델 학습
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 1206회 예측
model.eval()
with torch.no_grad():
    # 가장 최근 10회차 데이터를 가져와서 다음 회차 예측
    last_window = scaled_data[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 6)
    last_window_tensor = torch.FloatTensor(last_window)
    prediction_scaled = model(last_window_tensor)
    
    # 정규화된 값을 다시 1~45 범위로 역변환
    prediction = scaler.inverse_transform(prediction_scaled.numpy())

# 결과 정리 (반올림 및 정렬)
final_numbers = sorted(np.round(prediction[0]).astype(int))
# 중복 제거 및 범위 제한(1~45)
final_numbers = sorted(list(set([max(1, min(45, x)) for x in final_numbers])))

# 6개 번호 강제 맞춤
while len(final_numbers) < 6:
    extra = np.random.randint(1, 46)
    if extra not in final_numbers:
        final_numbers.append(extra)
        final_numbers.sort()

print(f"\n--- 1206회차 LSTM 예측 결과 ---")
print(f"예측 번호: {final_numbers}")

actual_1206 = sorted(df[df[round_col] == 1206][no_cols].values[0].tolist())
print(f"실제 번호: {actual_1206}")