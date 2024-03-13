import tensorflow as tf
tf.set_random_seed(777)

#1. data
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2. model
hypothesis = x * w + b

#3-1 compile
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
# model.compile(loss='mse', optimizer='sgd')

#3-2 train
sess = tf.compat.v1.Session()

sess.run(tf.global_variables_initializer())

# model.fit
epochs = 3000
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))
    
sess.close()
