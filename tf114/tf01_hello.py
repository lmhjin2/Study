import tensorflow as tf
# print(tf.__version__)  # 1.14.0

print('텐서플로우로 hello world')

# tf.variable, tf.constant, tf.placeholder  <- 텐서할때 꼭 알아야 할 것들
# tf.variable, tf.constant, tf.placeholder  <- 텐서할때 꼭 알아야 할 것들
# tf.variable, tf.constant, tf.placeholder  <- 텐서할때 꼭 알아야 할 것들

hello = tf.constant('hello world')
print(hello)

sess = tf.Session()
print(sess.run(hello))

bb = tf.constant('bb')
print(sess.run(bb))

sess.close()