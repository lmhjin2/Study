import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_iris

#1 data
datasets = load_iris()
x, y = datasets.data, datasets.target
# print(x.shape, y.shape)  # (150, 4) (150,)
x = x[ y != 2]  # y가 2인거 빼고 갖고옴
y = y[ y != 2]  # y가 2인거 빼고 갖고옴
# print(y, y.shape)   # (100,)

y = y.reshape(-1,1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,1], dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32))

#2
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(xp,w)+b)

#3-1
loss = -tf.reduce_mean( yp*tf.log(hypothesis) + (1-yp)*tf.log(1-hypothesis) )
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)

#3-2
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 50001
for step in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={xp:x, yp:y})
    if step % 100 == 0:
        print(step, loss_val)

#4
x_test = tf.compat.v1.placeholder(tf.float32, shape = [None,4])

y_pred = tf.compat.v1.sigmoid(tf.matmul(x_test, w_val) + b_val)
y_predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32), feed_dict={x_test:x})
# print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y, y_predict)
print(f'ACC : {acc}')

sess.close()


