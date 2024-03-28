import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1 data
x, y = fetch_california_housing(return_X_y=True)
# print(x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=777)
# print(y_train.shape) # (16512,)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
# print(y_train.shape) # (16512,1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None,8])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1]))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]))

#2 model
hypothesis = tf.matmul(xp, w) + b

#3-1 compile
loss = tf.reduce_mean(tf.square(hypothesis - yp))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

#3-2
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=30001
for step in range(epochs):
    _, val_loss = sess.run([train, loss], feed_dict={xp:x_train, yp:y_train})
    if step % 100 == 0:
        print(step, val_loss)

y_predict = sess.run(hypothesis, feed_dict={xp:x_test})
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f'MSE : {mse} \nR2 : {r2}')

sess.close()

# MSE : 0.6171168638971158
# R2 : 0.5212619805111574



