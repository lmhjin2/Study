import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1 데이터
x,y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape) # (442, 10) / (442,)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=777)
# print(y_train.shape) (353,)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# print(y_train.shape) (353,1)

#2
x = tf.compat.v1.placeholder(tf.float32, shape = [None,10])
y = tf.compat.v1.placeholder(tf.float32, shape = [None,1])
#layer1 : model.add(Dense(10, input_dim=2))
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,64], name = 'weight1'))
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([64], name = 'bias1' ))
layer1 = tf.compat.v1.matmul(x,w1) + b1         # (N, 64)
#layer2 : model.add(Dense(9))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,32], name = 'weight2'))
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([32], name = 'bias2' ))
layer2 = tf.compat.v1.matmul(layer1,w2) + b2    # (N, 32)
#layer3 : model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,16], name = 'weight3'))
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([16], name = 'bias3' ))
layer3 = tf.compat.v1.matmul(layer2,w3) + b3    # (N, 16)
#layer4 : model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,8], name = 'weight4'))
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([8], name = 'bias4' ))
layer4 = tf.nn.swish(tf.compat.v1.matmul(layer3,w4) + b4)    # (N, 8)
#output_layer : model.add(dense(1), activation='sigmoid')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1], name = 'weight5'))
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name = 'bias5' ))
hypothesis = tf.nn.swish(tf.compat.v1.matmul(layer4,w5) + b5) # (N,1)


#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 30001
for step in range(epochs):
    _, val_loss = sess.run([train, loss], feed_dict={x:x_train, y:y_train})
    if step % 1000 == 0:
        print(step, val_loss)

y_predict = sess.run(hypothesis, feed_dict={x:x_test})
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f'MSE : {mse} \nR2 : {r2}')

sess.close()

# MSE : 2679.227504127932
# R2 : 0.5110515730417939