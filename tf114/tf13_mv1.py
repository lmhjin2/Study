import tensorflow as tf
tf.compat.v1.set_random_seed(9810)
#1. 데이터
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

# w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], dtype=tf.float32, name='weight1'))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], dtype=tf.float32, name='weight2'))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], dtype=tf.float32, name='weight3'))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], dtype=tf.float32, name='bias'))

# hypothesis = x1 * x2 * x3 * w + b
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

loss = tf.reduce_mean(tf.square(hypothesis-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5) # 0.00001
# optimizer=tf.train.AdamOptimizer(learning_rate=0.0001)
train=optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
### 내꺼
# for step in range(epochs):
#     _, loss_v , w1_val, w2_val, w3_val, b_val = sess.run([train, loss, w1, w2, w3, b], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
#     if step % 100 == 0:
#         print(step, '\t', loss_v, '\t', w1_val, w2_val, w3_val, '\t', b_val)
    
### 쌤꺼
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if step % 20 == 0:
        print(step, cost_val)


sess.close()

