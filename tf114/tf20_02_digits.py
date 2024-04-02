import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#1 data
datasets = load_digits()
x_data = datasets.data
y_data = datasets.target
# print(x.shape, y.shape)  # (1797, 64) , (1797,)
# print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
y_data = y_data.reshape(-1,1)

encoder = OneHotEncoder(sparse=False)
y_data = encoder.fit_transform(y_data)

scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
# print(y_data.shape)  # (1797, 10)

#2
x = tf.compat.v1.placeholder(tf.float32, shape = [None,64])
y = tf.compat.v1.placeholder(tf.float32, shape = [None,10])
#layer1 : model.add(Dense(10, input_dim=2))
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,64], name = 'weight1'))
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
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,10], name = 'weight4'))
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], name = 'bias4' ))
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3,w4) + b4)    # (N, 10)
#output_layer : model.add(dense(1), activation='sigmoid')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,10], name = 'weight5'))
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], name = 'bias5' ))
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4,w5) + b5) # (N,10)

# 3-1. compile
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis + 1e-5 ),axis=1))  #categorical
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4)
# train = optimizer.minimize(loss)
# ===똑같음
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=4e-1).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 20001
for step in range(epochs):
    loss_val, _ = sess.run([loss,train],
                           feed_dict={x:x_data, y:y_data})
    if step %100 == 0:
        print(step, "loss : ", loss_val)


pred = sess.run(hypothesis, feed_dict={x:x_data})
# print(pred) 
pred = sess.run(tf.argmax(pred,axis=1))
# print(pred)
y_data = np.argmax(y_data, axis=1)
# print(y_data)

acc = accuracy_score(y_data,pred)
print("acc : ", acc)

sess.close()

# 20000 loss :  0.2307978
# acc :  0.9232053422370617