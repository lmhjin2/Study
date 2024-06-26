import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

path = "c:/_data/dacon/wine/"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":0, "red":1})
test_csv['type'] = test_csv['type'].replace({"white":0, "red":1})

x_data = train_csv.drop(['quality'], axis = 1)
y_data = train_csv['quality']
# print(x_data.shape, y_data.shape) # (5497, 12) (5497,)
# print(np.unique(y_data, return_counts=True))  
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64),          # 라벨 7개 
# array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
# test_csv = scaler.transform(test_csv)

y_data = y_data.values.reshape(-1, 1) # (5497, 1)
ohe = OneHotEncoder(sparse=False).fit(y_data)
y_data = ohe.transform(y_data)  # (5497, 7)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,12])
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([12,7]), name='weight')   
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1,7]), name='bias') 
y = tf.compat.v1.placeholder(tf.float32, shape=[None,7])

#2
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x,w) + b)

# 3-1. compile
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis + 1e-7 ),axis=1))  #categorical

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4)
# train = optimizer.minimize(loss)
# ===똑같음
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

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

# acc :  0.5472075677642351
