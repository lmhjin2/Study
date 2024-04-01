import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#1. 데이터
datasets = fetch_covtype()
x_data = datasets.data
y_data = datasets.target
# print(x.shape, y.shape)  # (581012, 54) (581012,)
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))  라벨 = 7개
y_data = y_data.reshape(-1,1) # (581012, 1)

ohe = OneHotEncoder(sparse=False)
y_data = ohe.fit_transform(y_data) # (581012, 7)

scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,54])
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([54,7]), name='weight')   
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1,7]), name='bias') 
y = tf.compat.v1.placeholder(tf.float32, shape=[None,7])

#2
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x,w) + b)

# 3-1. compile
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis + 1e-7 ),axis=1))  #categorical

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4)
# train = optimizer.minimize(loss)
# ===똑같음
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 20001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss,train,w,b],
                                         feed_dict={x:x_data, y:y_data})
    if step %100 == 0:
        print(step, "loss : ", cost_val)


pred = sess.run(hypothesis, feed_dict={x:x_data})
# print(pred) 
pred = sess.run(tf.argmax(pred,axis=1))
# print(pred)
y_data = np.argmax(y_data, axis=1)
# print(y_data)

acc = accuracy_score(y_data,pred)
print("acc : ", acc)

sess.close()


