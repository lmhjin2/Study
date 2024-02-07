import numpy as np
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

#1 데이터
x_data = np.array([[0,0],[0,1],[1,0],[1,1]]) # (4,2)
y_data = np.array([0, 1, 1, 0]) # (4,)

#2 모델
# model = LinearSVC()
# model = Perceptron()
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metr2ics=['acc'])
model.fit(x_data,y_data, batch_size=1, epochs=100)

#4 평가 예측
result = model.evaluate(x_data,y_data)
y_predict = model.predict(x_data)

y_predict = np.round(y_predict).reshape(-1,).astype(int)
accuracy = accuracy_score(y_data, y_predict)

print('loss', result[0])
print('acc', result[1])
print('accuracy', accuracy)
print('y_data',y_data)
print('y_predict',y_predict)

# loss 0.6146984100341797
# acc 1.0
# accuracy 1.0
# y_data [0 1 1 0]
# y_predict [0 1 1 0]


