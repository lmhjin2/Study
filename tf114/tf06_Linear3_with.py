import tensorflow as tf
tf.set_random_seed(777)

#1. data
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2. model
hypothesis = x * w + b
# ★ y = xw + b ★

#3-1 compile
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
# model.compile(loss='mse', optimizer='sgd') stochastic gradient descent

#3-2 train
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # model.fit
    epochs = 3000
    for step in range(epochs):
        sess.run(train)
        if step % 20 == 0:  # verbose
            print(step, sess.run(loss), sess.run(w), sess.run(b))
        
    # sess.close()
# 이렇게 하면 sess.close()를 안해도 알아서 닫힘.


