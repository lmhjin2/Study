import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. data
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]  # (6,2)
y_data = [[0], [0], [0], [1], [1], [1]]              # (6,1)

#####################################################
### [실습] 그냥한번 만들어봐
#####################################################

# x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
# y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1], name='weights'))
# b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias'))
# #2. model
# hypothesis = tf.matmul(x,w) + b
# #3-1. compile
# loss = tf.reduce_mean(tf.square(hypothesis - y))
# optimizer = tf.train.AdamOptimizer(learning_rate=1)
# train = optimizer.minimize(loss)

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# #3-2. train
# epochs = 30001
# for step in range(epochs):
#     _, val_loss = sess.run([train, loss], feed_dict={x:x_data, y:y_data})
#     if step % 100 == 0:
#         print(step, val_loss)

#####################################################
### [쌤코드]
#####################################################
#1. data
x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1], dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32))
#2. model
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w) + b)

#3-1. compile
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis-y))

loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))
#                   y의 실제값이 1인거 계산 // 실제값 0인거 계산.  
#                      실제값이 0이면 죽음  // 실제값이 1이면 죽음.

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)

#3-2 train
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 5001
for step in range(epochs):
    loss_val, _, w_val, b_val = sess.run([loss, train, w, b],
                                         feed_dict={x:x_data, y:y_data})
    if step % 300 == 0:
        print(step, "loss : ", loss_val)

# print(w_val, b_val)
# print(type(w_val))  # <class 'numpy.ndarray'>
## sess.run 을 통과해서 나온 값은 모두 'numpy' 형태다

#4 평가 예측. evaluate predict
x_test = tf.compat.v1.placeholder(tf.float32, shape = [None,2])

# y_predict = x_test * w_val + b_val
y_pred = tf.compat.v1.sigmoid(tf.matmul(x_test, w_val) + b_val)
y_predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32), feed_dict={x_test:x_data})
#                    < 반올림 하는 부분 >
print(y_predict)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

acc = accuracy_score(y_data, y_predict)
print("ACC : ", acc)


sess.close()