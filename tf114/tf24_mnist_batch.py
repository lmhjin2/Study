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

learning_rate = 1e-3
# [실습]

# 2
x = tf.compat.v1.placeholder(tf.float32, shape = [None,784])
y = tf.compat.v1.placeholder(tf.float32, shape = [None,10])

keep_prob = tf.compat.v1.placeholder(tf.float32)
# 가중치 초기화 = 임의의 초기 값으로 변수들 지정
#layer1 : model.add(Dense(128, input_dim=784))
# w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([784,128], name = 'weight1'))
w1 = tf.compat.v1.get_variable('w1', shape=[784,128],   # random_normal 이미 포함
                               initializer=tf.contrib.layers.xavier_initializer()) 
        # xavier = 가중치 초기화 기법. 이거 말고도 많은데 이거 쓸만함.           
        # get_variable 안에 if 문이 들어있어서 initializer는 최초에 한번(1epoch때)만 적용됨.
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([128], name = 'bias1' ))
layer1 = tf.compat.v1.matmul(x,w1) + b1         # (N, 128)
layer1 = tf.compat.v1.nn.relu(layer1)
layer1 = tf.compat.v1.nn.dropout(layer1, keep_prob=keep_prob)

#layer2 : model.add(Dense(32))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([128,64], name = 'weight2'))
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([64], name = 'bias2' ))
layer2 = tf.compat.v1.matmul(layer1,w2) + b2    # (N, 64)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)  # tf : keep_prob / keras : rate // 의미 역전 주의

#layer3 : model.add(Dense(16))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,32], name = 'weight3'))
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([32], name = 'bias3' ))
layer3 = tf.compat.v1.matmul(layer2,w3) + b3    # (N, 32)
layer3 = tf.compat.v1.nn.relu(layer3)
layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

#layer4 : model.add(Dense(10))
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,16], name = 'weight4'))
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([16], name = 'bias4' ))
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3,w4) + b4)    # (N, 16)

#output_layer : model.add(dense(16), activation='sigmoid')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,10], name = 'weight5'))
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], name = 'bias5' ))
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4,w5) + b5) # (N,10)


# 3-1. compile
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis + 1e-7 ),axis=1))  #categorical
# loss = tf.reduce_mean(-tf.reduce_sum( y * tf.compat.v1.nn.log_softmax(hypothesis), axis =1))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4)
# train = optimizer.minimize(loss)
# ===똑같음

train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 100
total_batch = int(len(x_train) / batch_size)
# 60000 / 100


training_epochs = 2001
for step in range(training_epochs):
    avg_loss = 0    # loss == cost
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y, keep_prob:0.7}
        
        loss_val, _ = sess.run([loss,train], feed_dict=feed_dict)
        
        avg_loss += loss_val / total_batch
        
        
    if step %10 == 0:
        print(step, "loss : ", avg_loss)


pred = sess.run(hypothesis, feed_dict={x:x_test, keep_prob:1.0})
# print(pred) 
pred = sess.run(tf.argmax(pred,axis=1))
# print(pred)
y_data = np.argmax(y_test, axis=1)
# print(y_data)

acc = accuracy_score(y_data,pred)
print("acc : ", acc)

sess.close()


# 5000 loss :  0.4394274
# acc :  0.9343

# dropout
# 5000 loss :  0.2932881
# acc :  0.9479

# initializer
# 5000 loss :  0.018009841
# acc :  0.9708

# 2000 loss :  0.2113182081095873
# acc :  0.9672