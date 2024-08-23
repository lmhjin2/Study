import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, F1Score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score

# print(torch.__version__)
# 2.2.2+cu118  

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print(f'torch : {torch.__version__}, 사용DEVICE : {DEVICE}')
# torch : 1.12.1, 사용DEVICE : cuda

#1. 데이터 
path = "c:/_data/kaggle/bike/"
train_csv = pd.read_csv(path +"train.csv", index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sampleSubmission.csv")

x = train_csv.drop(['casual', 'registered', 'count'], axis=1).to_numpy()
y = train_csv['count'].to_numpy()

print(x.shape, y.shape)  # (10886, 8) (10886,) // torchTensor 는 .shape가 파란색, numpy는 하얀색
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, shuffle=True) 

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 'Series' 계열 못알아 들어서 .to_numpy() 해줘야함
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape)  # (9253, 8) (9253, 1) // torchTensor 는 .shape가 파란색, numpy는 하얀색
from torch.utils.data import DataLoader, random_split, TensorDataset
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=2000, shuffle=True)
test_loader = DataLoader(test_set, batch_size=2000, shuffle=False)

#2. 모델구성

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 16)
        self.linear6 = nn.Linear(16, 8)
        self.linear7 = nn.Linear(8, 4)
        self.linear8 = nn.Linear(4, output_dim)
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
        x = self.bn(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear6(x)
        x = self.drop(x)
        x = self.linear7(x)
        x = self.relu(x)
        x = self.linear8(x)
        return x

model = Model(8, 1).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()                #criterion : 표준
optimizer = optim.Adam(model.parameters(), lr = 0.05)

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

epochs = 5000
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

# y_pred = np.round(model(x_test).cpu().detach().numpy())
y_pred = model(x_test).cpu().detach().numpy()
loss2 = evaluate(model, criterion, test_loader)
score = r2_score(y_test.cpu().numpy(), y_pred)
print(f"최종 loss : {loss2}")
print(f'rmse : {np.sqrt(loss2)}')
print(f'r2 : {score}')
