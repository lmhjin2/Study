import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, F1Score
from torch.utils.data import DataLoader, random_split, TensorDataset

# print(torch.__version__)
# 2.2.2+cu118

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print(f'torch : {torch.__version__}, 사용DEVICE : {DEVICE}')
# torch : 1.12.1, 사용DEVICE : cuda

#1. 데이터 
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

# x = torch.FloatTensor(x).to(DEVICE)  
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

# print(x.shape, y.shape)  # (569, 30) (569,) // torchTensor 는 .shape가 파란색, numpy는 하얀색
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y, shuffle=True) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

# print(x_train.shape, y_train.shape)  # (445, 30) (445, 1) // torchTensor 는 .shape가 파란색, numpy는 하얀색
#2. 모델구성
model = nn.Sequential(
    nn.Linear(30, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 7),
    nn.ReLU(),
    nn.Linear(7, 1),
    nn.Sigmoid()
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss()                #criterion : 표준
optimizer = optim.Adam(model.parameters(), lr = 0.01)
# optimizer = optim.SGD(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()   # 훈련모드 default / (dropout, normalization 등) O
    # model.eval()    # 평가모드 / (dropout, normalization 등) X
    
    optimizer.zero_grad()
    # w = w - lr * (loss를 weight로 미분한 값)
    hypothesis = model(x) #예상치 값 (순전파)
    loss = criterion(hypothesis, y) #예상값과 실제값 loss
    
    #역전파
    loss.backward() #기울기(gradient) 계산 (loss를 weight로 미분한 값)
    optimizer.step() # 가중치(w) 수정(weight 갱신)
    return loss.item() #item 하면 numpy 데이터로 나옴

epochs = 7000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    # print('epoch : {}, loss : {}'.format(epoch, loss))  # verbose
    print(f'epoch : {epoch}, loss : {loss}')              # verbose 

print("="*50)

#4 평가, 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()  # 평가모드
    accuracy_metric = BinaryAccuracy().to(DEVICE)
    f1_metric = F1Score(task='binary').to(DEVICE) 
    # f1_metric = BinaryF1Score().to(DEVICE)      
    
    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_test, y_predict)
        
        y_pred_class = (y_predict >= 0.5).float()  # 0.5 이상이면 1, 아니면 0 / 반올림 // torchmetric쓰면 안써도 되긴함
        
        ## ACC 1
        # accuracy = (y_pred_class == y_test).float().mean()
        
        ## ACC 2
        # accuracy = accuracy_metric(y_pred_class, y_test)
        accuracy = accuracy_metric(y_predict, y_test)
        
        ## F1
        # f1 = f1_metric(y_pred_class, y_test)  
        f1 = f1_metric(y_predict, y_test)

    return loss2.item(), accuracy.item(), f1.item()

loss2, accuracy, f1 = evaluate(model, criterion, x_test, y_test)
print(f"최종 loss : {loss2}")
print(f"f1 : {f1}")         # % 는 그냥 출력되라고 쓰는거임. 기능 x
print(f"ACC : {accuracy}")  # :.2f = 소수점 아래 두자리 까지 표시.

y_pred = np.round(model(x_test).cpu().detach().numpy())
# print(y_pred)
score = accuracy_score(y_test.cpu().numpy(), y_pred)
print(f'accuracy : {score:}')

result = model(x_test[0])
print(f"x_test[0] 예측값 : {result.item()}")
