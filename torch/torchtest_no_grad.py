import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 모델 초기화
model = SimpleModel()

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
  
# 학습 데이터 예시
data1 = torch.randn(5, 10)  # 첫 번째 배치 입력 데이터
target1 = torch.randn(5, 1)  # 첫 번째 배치 타겟 데이터
data2 = torch.randn(5, 10)  # 두 번째 배치 입력 데이터
target2 = torch.randn(5, 1)  # 두 번째 배치 타겟 데이터

# 그래디언트 초기화하지 않은 경우
model.train() 

# 첫 번째 배치 처리
optimizer.zero_grad()
output1 = model(data1)
loss1 = criterion(output1, target1)
loss1.backward()

# 두 번째 배치 처리
output2 = model(data2)
loss2 = criterion(output2, target2)
loss2.backward()

print("Gradients after processing two batches without zero_grad:")
for param in model.parameters():
    print(param.grad)

# 그래디언트 초기화한 경우
model.train()

# 첫 번째 배치 처리
optimizer.zero_grad()
output1 = model(data1)
loss1 = criterion(output1, target1)
loss1.backward()

# 그래디언트 초기화
optimizer.zero_grad()

# 두 번째 배치 처리
output2 = model(data2)
loss2 = criterion(output2, target2)
loss2.backward()

print("\nGradients after processing two batches with zero_grad:")
for param in model.parameters():
    print(param.grad)

# Gradients after processing two batches without zero_grad:
# tensor([[ 0.8116,  0.6205, -1.3885,  0.5837,  3.5297, -0.0289,  2.1402, -0.1883,
#          -0.5418, -0.1165]])
# tensor([0.2271])

# Gradients after processing two batches with zero_grad:
# tensor([[ 0.6391, -1.2007, -0.5926, -0.1671,  1.8756, -0.3090,  1.6357, -0.6703,
#           0.7222,  0.7482]])
# tensor([1.2709])

# 값이 달라지는걸 볼 수 있음.