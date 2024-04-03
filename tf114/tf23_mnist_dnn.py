import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# 1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.  # 255. 아니면 127.5로 나눠서 정규화함.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.    # (60000, 784) / (10000, 784)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32')/255. 
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]).astype('float32')/255. 

learning_rate = 3e-1
# [실습]

# 2
x = tf.compat.v1.placeholder(tf.float32, shape = [None,784])
y = tf.compat.v1.placeholder(tf.float32, shape = [None,10])

keep_prob = tf.compat.v1.placeholder(tf.float32)

#layer1 : model.add(Dense(64, input_dim=784))
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([784,64], name = 'weight1'))
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([64], name = 'bias1' ))
layer1 = tf.compat.v1.matmul(x,w1) + b1         # (N, 64)
layer1 = tf.compat.v1.nn.relu(layer1)
layer1 = tf.compat.v1.nn.dropout(layer1, keep_prob=keep_prob)

#layer2 : model.add(Dense(32))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,32], name = 'weight2'))
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([32], name = 'bias2' ))
layer2 = tf.compat.v1.matmul(layer1,w2) + b2    # (N, 32)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

#layer3 : model.add(Dense(16))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,16], name = 'weight3'))
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([16], name = 'bias3' ))
layer3 = tf.compat.v1.matmul(layer2,w3) + b3    # (N, 16)
layer3 = tf.compat.v1.nn.relu(layer3)
layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

#layer4 : model.add(Dense(10))
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,10], name = 'weight4'))
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], name = 'bias4' ))
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3,w4) + b4)    # (N, 10)

#output_layer : model.add(dense(10), activation='sigmoid')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,10], name = 'weight5'))
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], name = 'bias5' ))
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4,w5) + b5) # (N,10)


# 3-1. compile
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis + 1e-7 ),axis=1))  #categorical
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4)
# train = optimizer.minimize(loss)
# ===똑같음
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 5001
for step in range(epochs):
    loss_val, _ = sess.run([loss,train],
                           feed_dict={x:x_train, y:y_train, keep_prob:0.8})
    if step %100 == 0:
        print(step, "loss : ", loss_val)


pred = sess.run(hypothesis, feed_dict={x:x_test, keep_prob:1.0})
# print(pred) 
pred = sess.run(tf.argmax(pred,axis=1))
# print(pred)
y_data = np.argmax(y_test, axis=1)
# print(y_data)

acc = accuracy_score(y_data,pred)
print("acc : ", acc)

sess.close()


# 3000 loss :  0.616116
# acc :  0.9141


