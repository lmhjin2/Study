import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]  # (6,2)
y_data = [[0], [0], [0], [1], [1], [1]]              # (6,1)

#####################################################
### [실습] 그냥한번 만들어봐
#####################################################

x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1], name='weights'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias'))

hypothesis = tf.matmul(x,w) + b

loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.AdamOptimizer(learning_rate=1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 30001
for step in range(epochs):
    _, val_loss = sess.run([train, loss], feed_dict={x:x_data, y:y_data})
    if step % 100 == 0:
        print(step, val_loss)

