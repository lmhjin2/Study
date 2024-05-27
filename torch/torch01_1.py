import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# print(torch.__version__)
# 2.2.2+cu118

#1. 데이터 
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1)  # reshape를 unsqueeze로 해준거임.
y = torch.FloatTensor(y).unsqueeze(1)  # x만 하고 y를 unsqueeze 안해주면 y의 평균값으로 수렴함 (2)

# print(x, y) # tensor([1., 2., 3.]) tensor([1., 2., 3.])
#             # ([[1.], [2.], [3.]]) , ([1., 2., 3.])
#             # unsequeeze.(1)       ,  기본

# print(x.shape, y.shape)  # ([3,1]) , ([3]) 
#                          # unsequeeze.(1), 기본

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
model = nn.Linear(1, 1) # input, output / 케라스랑 반대

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()    # criterion : 표준 / (h-y)^2 를 다 더하고 N빵 = MSE
# optimizer = optim.Adam(model.parameters(), lr = 0.01)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# model.fit(x,y, epochs = 100, batch_size=1)
def train(model, criterion, optimizer, x, y):
    # model.train()   # 훈련모드 default / (dropout, normalization 등) O
    # model.eval()    # 평가모드 / (dropout, normalization 등) X
    
    optimizer.zero_grad()  # gradient 0으로 초기화 / 배치당 1번 해줘야함
    # w = w - lr * (loss를 weight로 미분한 값 (=gradient))
    hypothesis = model(x) # 예측값 (순전파)  / y = xw + b
    loss = criterion(hypothesis, y) # 예측값과 실제값 loss
    
    #역전파
    loss.backward() # 기울기(gradient) 계산           / 역전파 시작
    optimizer.step() # 가중치(w) 수정(weight 갱신)   / 역전파 끝
    return loss.item() # item 하면 numpy 데이터로 나옴

epochs = 2200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss : {}'.format(epoch, loss))  # verbose
    print(f'epoch : {epoch}, loss : {loss}')

print("="*50)

#4 평가, 예측
# loss = model.evaluate(x_test,y_test)        
def evaluate(model, criterion, x, y):
    model.eval()  # 평가모드
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print("최종 loss : ", loss2)

# result = model.predict([4])
result = model(torch.Tensor([[4]]))
print(f"4의 예측값 : {result.item()}")

# ==================================================
# 최종 loss :  5.0026969233840646e-08
# 4의 예측값 : 4.0004496574401855



