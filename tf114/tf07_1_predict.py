import tensorflow as tf
tf.set_random_seed(777)

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
# model.compile(loss='mse', optimizer='sgd') stochastic gradient descent


#3-2 train
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # model.fit
    epochs = 3001
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 or step == epochs-1 :  
            print(step, loss_val, w_val, b_val)
        
    # sess.close()
# 이렇게 하면 sess.close()를 안해도 알아서 닫힘.

############ [실습] ####################
x_pred_data = [6,7,8]
# 예측값을 뽑아라
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # 가중치와 편향을 최신 상태로 업데이트합니다.
    sess.run([w.initializer, b.initializer])
    
    # 모델 학습
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x: x_data, y: y_data})
        # sess.run([train, loss, w, b], feed_dict={x: x_data, y: y_data})
        # 둘다 됨
    # 예측값 계산
    prediction = sess.run(hypothesis, feed_dict={x: x_pred_data})
    
    print("예측값:", prediction)

############################################################
    # 위쪽 with 코드에 포함되게끔 indent
    #1. 파이썬 방식
    y_predict = x_pred_data * w_val + b_val
    print('[6,7,8]의 예측 : ', y_predict)


    #2. placeholer에 넣어서
    y_predict2 = x_test * w_val + b_val
    print('[6,7,8]의 예측 : ', sess.run(y_predict2, feed_dict={x_test:x_pred_data}))






