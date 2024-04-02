import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1
x_data = [[0,0],[0,1], [1,0], [1,1]]  # (4,2)
y_data = [[0],[1],[1],[0]]            # (4,1)

x = tf.compat.v1.placeholder(tf.float32, shape = [None,2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1], name = 'weight'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias' ))

# [실습]
#2
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w) + b)

#3-1
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

#3-2 train
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
for step in range(epochs):
    loss_val, _, w_val, b_val = sess.run([loss, train, w, b],
                                         feed_dict={x:x_data, y:y_data})
    if step % 200 == 0:
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

