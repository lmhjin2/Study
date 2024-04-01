import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1 data
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
y_data = [[0,0,1], # 2
          [0,0,1],
          [0,0,1],
          [0,1,0], # 1
          [0,1,0],
          [0,1,0],
          [1,0,0], # 0
          [1,0,0]]

x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,4])
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,3])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal(dtype=tf.float32, shape=[4,3]))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1, 3]), name='bias')

#2 model
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

#3-1 compile
loss = tf.reduce_mean( -tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)
# 두줄에서 한줄로 만들기.
# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

#[실습]

#3-2 train
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 20001
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        if step % 100 == 0:
            print(step, loss_val)

with tf.compat.v1.Session() as sess : 
    sess.run(tf.compat.v1.global_variables_initializer())
    a = sess.run(hypothesis, feed_dict={x: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.arg_max(a, 1)))
    print('--------------------')
    b = sess.run(hypothesis, feed_dict={x: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.arg_max(b, 1)))
    print('--------------------')
    c = sess.run(hypothesis, feed_dict={x: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.arg_max(c, 1)))
    print('--------------------')
    # all = sess.run(hypothesis, feed_dict={x: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    all = sess.run(hypothesis, feed_dict={x:[
          [1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]})
    
    print(all, sess.run(tf.arg_max(all, 1)))






sess.close()
