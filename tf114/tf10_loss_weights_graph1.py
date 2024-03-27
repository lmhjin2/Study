import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

x = [1,2,3]
y = [1,2,3]
########### 손 계산 ###########################
# x = [1,2]
# y = [1,2]
# w = -30 부터 50 까지
# -30, 1, 50을 포함해서 아무숫자 8개
########### 손 계산 ###########################
w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis-y)) # == mse

w_history = []
loss_history=[]

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})

        w_history.append(curr_w)
        loss_history.append(curr_loss)

print("="*50, 'W history', "="*50)
print(w_history)
print("="*50, 'L history', "="*50)
print(loss_history)

plt.plot(w_history, loss_history)
plt.xlabel('Weights')
plt.ylabel('Loss')
plt.show()



