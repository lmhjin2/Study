import tensorflow as tf
tf.compat.v1.set_random_seed(777)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1
path = "c:/_data\\dacon/ddarung//"
train_csv = pd.read_csv(path+"train.csv", index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"submission.csv")

train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']  
# print(x.shape, y.shape) # (1328, 9) (1328,)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=777)
# print(y_train.shape) # (1062,)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
# print(y_train.shape) # (1062,1)
xp = tf.compat.v1.placeholder(tf.float32, shape=[None,9])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([9,1], name='weights'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias'))

#2 모델
hypothesis = tf.matmul(xp, w) + b

#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - yp))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 30001
for step in range(epochs):
    _, val_loss = sess.run([train, loss], feed_dict={xp:x_train, yp:y_train})
    if step % 100 == 0:
        print(step, val_loss)

y_predict = sess.run(hypothesis, feed_dict={xp:x_test})
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f'MSE : {mse} \nR2 : {r2}')

sess.close()


# MSE : 2571.9192566343854
# R2 : 0.6111239154151966
