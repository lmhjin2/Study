import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# print(torch.__version__)
# 2.2.2+cu118

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# print(f'torch : {torch.__version__}, 사용DEVICE : {DEVICE}')
# torch : 1.12.1, 사용DEVICE : cuda

#1. 데이터 
x = np.array([range(10), range(21,31), range(201, 211)]) # (0~9), (21~30), (201~210) / shape = (3, 10)
x = x.transpose()
# x = np.transpose(x) # (3, 10) -> (10, 3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])
y = y.transpose()  # (3, 10) -> (10, 3)

x = torch.FloatTensor(x).to(DEVICE)  # unsqueeze하면 (10, 1, 3) 되니까 없앰
y = torch.FloatTensor(y).to(DEVICE)  # (10, 1)
# print(x.shape)
# print(y.shape)

x_mean = torch.mean(x)
x_std = torch.std(x)
x = (x - x_mean) / x_std

#2. 모델구성
# model = nn.Linear(1, 1).to(DEVICE) # input, output / 케라스랑 반대
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 3),
    nn.Linear(3, 3)
).to(DEVICE)


#3. 컴파일, 훈련
criterion = nn.MSELoss()                #criterion : 표준
optimizer = optim.Adam(model.parameters(), lr = 0.01)

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

epochs = 5000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss : {}'.format(epoch, loss))  # verbose
    print(f'epoch : {epoch}, loss : {loss}')

print("="*50)

#4 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model, criterion, x, y):
    model.eval()  # 평가모드
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print("최종 loss : ", loss2)

input_value = torch.FloatTensor([[10, 31, 211]]).to(DEVICE)
input_value = (input_value - x_mean) / x_std  # 동일한 스케일링 적용
result = model(input_value)
# 1)
result_list = result.detach().cpu().numpy().tolist()
# print(f"[[10, 31, 211]]의 예측값 : {result_list}")
# ==================================================
# 최종 loss :  3.6110691260110572e-12
# [[10, 31, 211]]의 예측값 : [[10.999998092651367, 1.9999996423721313, -0.9999979138374329]]
# 2)
for i, res in enumerate(result[0], start=1):
    print(f'{i}번째 예측값 :', res.item())
# ==================================================
# 최종 loss :  1.197294216895295e-12
# 1번째 예측값 : 11.000000953674316
# 2번째 예측값 : 1.9999998807907104
# 3번째 예측값 : -1.0000001192092896

# 실제값 : [[11     2       -1]]
"""
formatted_result = [float(val) for val in result_list[0]]  # 1) 랑 똑같이 나옴
# formatted_result = [int(round(val)) for val in result_list[0]]  # 반올림
print(f"예측값 : {formatted_result}")
"""




