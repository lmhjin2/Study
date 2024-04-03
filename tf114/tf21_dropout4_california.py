import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1 data
x, y = fetch_california_housing(return_X_y=True)
# print(x.shape, y.shape) # (20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=777)
# print(y_train.shape) # (16512,)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
# print(y_train.shape) # (16512,1)

#2
x = tf.compat.v1.placeholder(tf.float32, shape = [None,8])
y = tf.compat.v1.placeholder(tf.float32, shape = [None,1])

keep_prob = tf.compat.v1.placeholder(tf.float32)

#layer1 : model.add(Dense(64, input_dim=8))
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,64], name = 'weight1'))
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([64], name = 'bias1' ))
layer1 = tf.nn.relu(tf.compat.v1.matmul(x,w1) + b1)         # (N, 64)

#layer2 : model.add(Dense(32))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,32], name = 'weight2'))
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([32], name = 'bias2' ))
layer2 = tf.compat.v1.matmul(layer1,w2) + b2    # (N, 32)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

#layer3 : model.add(Dense(16))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,16], name = 'weight3'))
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([16], name = 'bias3' ))
layer3 = tf.compat.v1.matmul(layer2,w3) + b3    # (N, 16)

#layer4 : model.add(Dense(8))
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,8], name = 'weight4'))
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([8], name = 'bias4' ))
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3,w4) + b4)    # (N, 8)

#output_layer : model.add(dense(1), activation='sigmoid')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1], name = 'weight5'))
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias5' ))
hypothesis = tf.compat.v1.matmul(layer4,w5) + b5 # (N,1)

#3-1 compile
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)

#3-2
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=10001
for step in range(epochs):
    _, val_loss = sess.run([train, loss], feed_dict={x:x_train, y:y_train, keep_prob:0.7})
    if step % 100 == 0:
        print(step, val_loss)

y_predict = sess.run(hypothesis, feed_dict={x:x_test, keep_prob:1.0})
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f'MSE : {mse} \nR2 : {r2}')

sess.close()


# MSE : 1.2876434452235648
# R2 : 0.00109054080737081

