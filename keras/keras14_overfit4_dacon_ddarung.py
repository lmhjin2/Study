import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_squared_log_error

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

#1
path = "c:/_data\\dacon/ddarung//"
train_csv = pd.read_csv(path+"train.csv", index_col = 0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())
x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.85, random_state = 65456 )

#2
model = Sequential()
model.add(Dense(18, input_dim = 9))
model.add(Dense(36))
model.add(Dense(53))
model.add(Dense(23))
model.add(Dense(12))
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=300, batch_size=50,
          validation_split=0.2, verbose=2)

#4
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

submission_csv['count'] = y_submit
submission_csv.to_csv(path+"submission_0109_plt.csv",
                      index = False)

print('loss:', loss)
print('r2:', r2)

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color = 'red',
         label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue',
         label = 'val_loss', marker = ',')
plt.legend(loc='upper right')
plt.title('따릉이 로스값')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

