import tensorflow as tf
tf.compat.v1.set_random_seed(777)
import matplotlib.pyplot as plt

#1. data
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
# print(sess.run(w), sess.run(b))

#2. model
hypothesis = x * w + b
# ★ y = xw + b ★

#3-1 compile
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08235)
train = optimizer.minimize(loss)
# model.compile(loss='mse', optimizer='sgd') stochastic gradient descent

# 100 0.0011131145 [2.0119038] [0.9912439]  0.0825
# 100 0.00068634364 [2.0099583] [0.9906625] 0.0824
# 100 0.0004264488 [2.0084443] [0.99020076] 0.0823


#3-2 train
loss_val_list = []
w_val_list = []
b_val_list = []
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # model.fit
    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 or step == epochs-1 :  
            print(step, loss_val, w_val, b_val)
        loss_val_list.append(loss_val)    
        w_val_list.append(w_val)
        b_val_list.append(b_val)
# history 리스트 찍어보기
print(loss_val_list)
print('')
print(w_val_list)

# val_loss 시각화
plt.subplot(2,2,1)
plt.plot(loss_val_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('val_loss visualize')

# weights 시각화
plt.subplot(2,2,2)
plt.plot(w_val_list)
plt.xlabel('epochs')
plt.ylabel('weights')
plt.title('weights visualize')

# bias 시각화
plt.subplot(2,2,3)
plt.scatter(range(len(b_val_list)), b_val_list)
plt.xlabel('epochs')
plt.ylabel('bias')
plt.title('bias visualize')

# val_loss와 weights의 관계 시각화
plt.subplot(2,2,4)
plt.scatter(w_val_list, loss_val_list)
plt.xlabel('weights')
plt.ylabel('loss')
plt.title('relations of val_loss weights visualize')

plt.tight_layout()
plt.show()

