# 02 digits
# 03 fetch_covtype
# 04 dacon_wine
# 05 dacon_대출
# 06 kaggle obesity
# criterion = nn.CrossEntropyLoss()
# torch.argmax()
# acc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits, fetch_covtype
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, F1Score, Accuracy
from torch.utils.data import DataLoader, random_split, TensorDataset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. 데이터
path = "c:/_data/dacon/dechul/"
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

le_work_period = LabelEncoder() 
le_work_period.fit(train_csv['근로기간'])
train_csv['근로기간'] = le_work_period.transform(train_csv['근로기간'])
test_csv['근로기간'] = le_work_period.transform(test_csv['근로기간'])

le_grade = LabelEncoder()
le_grade.fit(train_csv['대출등급'])
train_csv['대출등급'] = le_grade.transform(train_csv['대출등급'])

le_purpose = LabelEncoder()
test_csv.iloc[34486,7] = '이사'     # 결혼 -> 이사 로 임의로 바꿈
le_purpose.fit(train_csv['대출목적'])
train_csv['대출목적'] = le_purpose.transform(train_csv['대출목적'])
test_csv['대출목적'] = le_purpose.transform(test_csv['대출목적'])

le_own = LabelEncoder()
le_own.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = le_own.transform(train_csv['주택소유상태'])
test_csv['주택소유상태'] = le_own.transform(test_csv['주택소유상태'])

le_loan_period = LabelEncoder()
le_loan_period.fit(train_csv['대출기간'])
train_csv['대출기간'] = le_loan_period.transform(train_csv['대출기간'])
test_csv['대출기간'] = le_loan_period.transform(test_csv['대출기간'])

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

print(x.shape, y.shape)  # (96294, 13) (96294,)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y, shuffle=True) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape, y_train.shape)  # (77035, 13) (77035)
print(np.unique(y))  # 0 ~ 6

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size = 10000, shuffle=True)
test_loader = DataLoader(test_set, batch_size = 10000, shuffle=False)

#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(13, 256),   1
#     nn.ReLU(),
#     nn.Linear(256, 128),  2
#     nn.Dropout(0.2),
#     nn.Linear(128, 64),   3
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(64, 32),    4
#     nn.BatchNorm1d(32),
#     nn.Linear(32, 16),    5
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(16, 7)      6
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 16)
        self.linear6 = nn.Linear(16, output_dim)
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
        return x

model = Model(13, 7).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()                #criterion : 표준
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, loader):
    model.train()   # 훈련모드 default / (dropout, normalization 등) O
    # model.eval()    # 평가모드 / (dropout, normalization 등) X
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        # w = w - lr * (loss를 weight로 미분한 값)
        hypothesis = model(x_batch) # 예상치 값 (순전파)
        loss = criterion(hypothesis, y_batch) # 예상값과 실제값 loss
        
        #역전파
        loss.backward() #기울기(gradient) 계산 (loss를 weight로 미분한 값)
        optimizer.step() # 가중치(w) 수정(weight 갱신)
        total_loss += loss.item()
        
    return total_loss / len(loader)

epochs = 2000
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
    accuracy_metric = Accuracy(task='multiclass', num_classes=7).to(DEVICE)
    f1_metric = F1Score(task='multiclass', num_classes=7).to(DEVICE)  # 둘다 됨
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_predict = model(x_batch)
            loss2 = criterion(y_predict, y_batch)
            total_loss += loss2.item()
                        
            y_pred_class = torch.argmax(y_predict, dim=1)
            accuracy_metric.update(y_pred_class, y_batch)
            f1_metric.update(y_pred_class, y_batch)

    accuracy = accuracy_metric.compute()
    f1 = f1_metric.compute()
    
    return total_loss / len(loader), accuracy.item(), f1.item()

loss2, accuracy, f1 = evaluate(model, criterion, test_loader)
print(f"최종 loss : {loss2}")
print(f"f1 : {f1}")         
print(f"ACC : {accuracy}")  

# ==================================================
# 최종 loss : 0.4291548728942871
# f1 : 0.8464094996452332
# ACC : 0.8464094996452332