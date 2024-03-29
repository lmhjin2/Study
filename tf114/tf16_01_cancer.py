import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#1
# x, y = load_breast_cancer(return_X_y=True)
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape)  # (569, 30) (569,)

scaler = StandardScaler()
x = scaler.fit_transform(x)

y = y.reshape(-1,1)

xp = tf.compat.v1.placeholder(tf.float32, shape=[None,30])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1], dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32))

#2
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(xp,w)+b)
hypothesis_clipped = tf.clip_by_value(hypothesis, 1e-7, 1 - 1e-7)
#3-1
loss = -tf.reduce_mean( yp*tf.log(hypothesis_clipped) + (1-yp)*tf.log(1-hypothesis_clipped) )
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

#3-2
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 20001
for step in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={xp:x, yp:y})
    if step % 100 == 0:
        print(step, loss_val)

#4
x_test = tf.compat.v1.placeholder(tf.float32, shape = [None,30])

y_pred = tf.compat.v1.sigmoid(tf.matmul(x_test, w_val) + b_val)
y_predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32), feed_dict={x_test:x})
# print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y, y_predict)
print(f'ACC : {acc}')

sess.close()
