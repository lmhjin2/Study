import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# 데이터 로드 및 전처리
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=120, shuffle=True)
test_loader = DataLoader(test_set, batch_size=120, shuffle=False)

# 모델 구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear3(x)
        return x

model = Model(10, 1).to(DEVICE)

# 컴파일 및 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

epochs = 10000
best_epoch = 0
best_loss = float('inf')
best_model_weights = None

for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    
    if loss < best_loss : 
        best_loss = loss
        best_epoch = epoch
        best_model_weights = model.state_dict().copy()
        print(f'epoch : {best_epoch}, loss : {best_loss} weights saved ')
    
    if epoch % 100 == 0:
        print(f'epoch : {epoch}, loss : {loss}')              # verbose 

print("="*50)

if best_model_weights:
    model.load_state_dict(best_model_weights)
    print(f"Best model weights restored from epoch {best_epoch}")
    

# 평가 및 예측
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_predict = model(x_batch)
            loss = criterion(y_predict, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

loss2 = evaluate(model, criterion, test_loader)
y_pred = model(x_test).cpu().detach().numpy()
score = r2_score(y_test.cpu().numpy(), y_pred)
print(f"최종 loss : {loss2}")
print(f'rmse : {np.sqrt(loss2)}')
print(f'r2 : {score}')

# ==================================================
# Best model weights restored.
# 최종 loss : 2933.04052734375
# rmse : 54.157552819009005
# r2 : 0.44640307712592286
