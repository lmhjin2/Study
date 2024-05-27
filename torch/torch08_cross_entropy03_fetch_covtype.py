# 02 digits
# 03 fetch_covtype
# 04 dacon_wine
# 05 dacon_대출
# 06 kaggle obesity
# criterion = nn.CrossEntropyLoss()
# torch.argmax()
# acc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits, fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, F1Score, Accuracy
from torch.utils.data import DataLoader, random_split, TensorDataset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. 데이터 
dataset = fetch_covtype()
x = dataset.data
y = dataset.target - 1

print(x.shape, y.shape)  # (581012, 54) (581012,)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y, shuffle=True) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape, y_train.shape)  # (464809, 54) (464809)
# print(np.unique(y))  # 1 ~ 7

#2. 모델구성
model = nn.Sequential(
    nn.Linear(54, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(16, 7)
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()                #criterion : 표준
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()   # 훈련모드 default / (dropout, normalization 등) O
    # model.eval()    # 평가모드 / (dropout, normalization 등) X
    
    optimizer.zero_grad()
    # w = w - lr * (loss를 weight로 미분한 값)
    hypothesis = model(x) # 예상치 값 (순전파)
    loss = criterion(hypothesis, y) # 예상값과 실제값 loss
    
    #역전파
    loss.backward() #기울기(gradient) 계산 (loss를 weight로 미분한 값)
    optimizer.step() # 가중치(w) 수정(weight 갱신)
    return loss.item() # item 하면 numpy 데이터로 나옴

epochs = 1000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    # print('epoch : {}, loss : {}'.format(epoch, loss))  # verbose
    print(f'epoch : {epoch}, loss : {loss}')              # verbose 

print("="*50)

#4 평가, 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()  # 평가모드
    accuracy_metric = Accuracy(task='multiclass', num_classes=7).to(DEVICE)
    f1_metric = F1Score(task='multiclass', num_classes=7).to(DEVICE)  # 둘다 됨
    
    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_predict, y_test)
        
        y_pred_class = torch.argmax(y_predict, dim=1)
        # ACC 
        accuracy = accuracy_metric(y_pred_class, y_test)
        
        # F1
        f1 = f1_metric(y_pred_class, y_test)

    return loss2.item(), accuracy.item(), f1.item()

loss2, accuracy, f1 = evaluate(model, criterion, x_test, y_test)
print(f"최종 loss : {loss2}")
print(f"f1 : {f1}")         
print(f"ACC : {accuracy}")  

# ==================================================
# 최종 loss : 0.2738306224346161
# f1 : 0.8865864276885986
# ACC : 0.8865864276885986
