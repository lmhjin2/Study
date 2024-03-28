import tensorflow as tf
tf.compat.v1.set_random_seed(9810)

#1 data
x_data = [[73,51,65],   # (5,3)
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]]
y_data = [[152], [185], [180],[205],[142]] # (5,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1], dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], dtype=tf.float32))
# x * w = y
# (N, 3) * (3, 1) = (N, 1)

#2 model
hypothesis = tf.matmul(x,w) + b

#3-1 compile
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=8e-5)
train = optimizer.minimize(loss)

#3-2 train
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    _, loss_val = sess.run([train, loss], feed_dict={x:x_data, y:y_data})
    if step % 100 == 0: # step 을 100으로 나눴을때 나머지가 0이라면
        print(step, loss_val)

