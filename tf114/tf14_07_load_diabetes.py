import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1 데이터
x,y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape) # (442, 10) / (442,)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=777)
# print(y_train.shape) (353,)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# print(y_train.shape) (353,1)
xp = tf.compat.v1.placeholder(tf.float32, shape=[None,10])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1], name='weights'))
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

# MSE : 2395.558580013047
# R2 : 0.5628200301844604