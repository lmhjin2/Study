# 07 load_diabetes
# 08 california
# 09 dacon ddarung
# 10 kaggle bike
# 평가 rmse, r2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, F1Score

# print(torch.__version__)
# 2.2.2+cu118

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print(f'torch : {torch.__version__}, 사용DEVICE : {DEVICE}')
# torch : 1.12.1, 사용DEVICE : cuda

#1. 데이터 
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (442, 10) (442,) // torchTensor 는 .shape가 파란색, numpy는 하얀색
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape)  # (353, 10) (353, 1) // torchTensor 는 .shape가 파란색, numpy는 하얀색
from torch.utils.data import DataLoader, random_split, TensorDataset

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(10, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.Dropout(0.2),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(32, 16),
#     nn.Linear(16, 8),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(8, 1)
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, 8)
        self.linear6 = nn.Linear(8, output_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.bn = nn.BatchNorm1d(32)
        return
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.drop(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear6(x)
        return x

model = Model(10, 1).to(DEVICE)

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()                #criterion : 표준
optimizer = optim.Adam(model.parameters())
# optimizer = optim.SGD(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, loader):
    model.train()   # 훈련모드 default / (dropout, normalization 등) O
    # model.eval()    # 평가모드 / (dropout, normalization 등) X
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        # w = w - lr * (loss를 weight로 미분한 값)
        hypothesis = model(x_batch) #예상치 값 (순전파)
        loss = criterion(hypothesis, y_batch) #예상값과 실제값 loss
        
        #역전파
        loss.backward() #기울기(gradient) 계산 (loss를 weight로 미분한 값)
        optimizer.step() # 가중치(w) 수정(weight 갱신)
        total_loss += loss.item()
        
    return total_loss / len(loader)

epochs = 10000
best_loss = float('inf')
best_model_weights = None

for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    
    if loss < best_loss : 
        best_loss = loss
        best_model_weights = model.state_dict().copy()
        print(f'epoch : {epoch}, loss : {best_loss} weights saved ')
    
    if epoch % 100 == 0:
        print(f'epoch : {epoch}, loss : {loss}')              # verbose 

print("="*50)

if best_model_weights:
    model.load_state_dict(best_model_weights)
    print("Best model weights restored.")

#4 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()  # 평가모드
    total_loss = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_predict = model(x_batch)
            loss2 = criterion(y_predict, y_batch)
            total_loss += loss2.item()
            
    return total_loss / len(loader)

loss2 = evaluate(model, criterion, test_loader)
y_pred = model(x_test).cpu().detach().numpy()
score = r2_score(y_test.cpu().numpy(), y_pred)
print(f"최종 loss : {loss2}")
print(f'rmse : {np.sqrt(loss2)}')
print(f'r2 : {score}')

# ==================================================
# 최종 loss : 3819.849853515625
# rmse : 61.804933892979975
# r2 : 0.2615317485143447
