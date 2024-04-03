import tensorflow as tf
tf.compat.v1.set_random_seed(777)
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

#1
datasets = load_breast_cancer()

x_data = datasets.data
y_data = datasets.target

scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

y_data = y_data.reshape(-1,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

keep_prob = tf.placeholder(tf.float32)

#2
# layer1 : model.add(Dense(32, input_dim = 30 ))
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,32], dtype=tf.float32, name='weight1'))
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([32], dtype=tf.float32, name='bias1'))
layer1 = tf.compat.v1.matmul(x,w1) + b1     # (N, 32)

# layer1 : model.add(Dense(24))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,24], dtype=tf.float32, name='weight2'))
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([24], dtype=tf.float32, name='bias2'))
layer2 = tf.compat.v1.matmul(layer1,w2) + b2     # (N, 24)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

# layer1 : model.add(Dense(16))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([24,16], dtype=tf.float32, name='weight3'))
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([16], dtype=tf.float32, name='bias3'))
layer3 = tf.compat.v1.matmul(layer2,w3) + b3     # (N, 16)
layer3 = tf.nn.dropout(layer3, keep_prob=0.5)

# layer1 : model.add(Dense(8))
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,8], dtype=tf.float32, name='weight4'))
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([8], dtype=tf.float32, name='bias4'))
layer4 = tf.compat.v1.matmul(layer3,w4) + b4     # (N, 8)

# layer1 : model.add(Dense(1))
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1], dtype=tf.float32, name='weight5'))
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32, name='bias5'))
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4,w5) + b5) # (N, 1)

hypothesis_clipped = tf.clip_by_value(hypothesis, 1e-7, 1 - 1e-7)

#3-1
loss = -tf.reduce_mean( y*tf.log(hypothesis_clipped) + (1-y)*tf.log(1-hypothesis_clipped) )
optimizer = tf.train.AdamOptimizer(learning_rate=5e-4)
train = optimizer.minimize(loss)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

#3-2
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#3-2 train
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(20001):
        loss_val, _ = sess.run([loss, train], feed_dict={x:x_data, y:y_data, keep_prob:0.5})
        
        if step % 200 == 0:
            print(step, loss_val)
    
    hypo, pred, acc = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={x:x_data, y:y_data, keep_prob:1.0})
    # print(f'훈련값 : {hypo}\n')
    # print(f'예측값 : {pred}\n')
    print(f'acc : {acc}\n')

# acc : 0.9314587116241455

# dropout
# acc : 0.9912126660346985