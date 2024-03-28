import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.set_random_seed(777)

#1. 데이터
x_train = [1,2,3]
y_train = [1,2,3]
x_test = [4,5,6]
y_test = [4,5,6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')
# b = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')

#2. 모델
# hypothesis = x * w + b
hypothesis = x * w 

#3-1. 컴파일 // model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis-y)) # == mse

###################### 옵티마이저 ###################################
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08235)
# train = optimizer.minimize(loss)

lr = 0.1
# gradient = tf.reduce_mean((x * w + b - y) * x)
gradient = tf.reduce_mean((x * w - y) * x)
descent = w - lr * gradient
update = w.assign(descent)
# w - lr * (( x * w - y) * x )
# w - lr * (loss를 w로 미분한 값)
###################### 옵티마이저 ###################################

w_history = []
loss_history=[]

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(37):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
        
    w_history.append(w_v)
    loss_history.append(loss_v)

################ R2, mae 만들기
from sklearn.metrics import r2_score, mean_absolute_error

y_pred = sess.run(hypothesis, feed_dict={x: x_test})

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(y_pred)
print("R2 : ", r2)
print("MAE : ", mae)

sess.close()






