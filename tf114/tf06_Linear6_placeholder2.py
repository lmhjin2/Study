import tensorflow as tf
tf.set_random_seed(777)

#1. data
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
# print(sess.run(w), sess.run(b))

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
    epochs = 3001
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 or step == epochs-1 :  
            print(step, loss_val, w_val, b_val)
        
    # sess.close()
# 이렇게 하면 sess.close()를 안해도 알아서 닫힘.
