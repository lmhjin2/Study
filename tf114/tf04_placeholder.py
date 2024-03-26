import tensorflow as tf
# print(tf.__version__)     # 1.14.0
# print(tf.executing_eagerly())     # False

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()

node1 = tf.constant(30.0, tf.float32) # 못바꿈 / variable은 바꿀 수 있음
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
# print(sess.run(node3))
# 34.0

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
add_node = a + b
# {a:3, b:4} 이거만 써도댐
print(sess.run(add_node, feed_dict={a:3, b:4}))     # 7.0
print(sess.run(add_node, feed_dict={a:30, b:4.5}))  # 34.5

add_and_triple = add_node *3
print(sess.run(add_and_triple, feed_dict={a:30, b:4.5}))

sess.close()

