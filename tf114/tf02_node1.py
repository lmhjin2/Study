import tensorflow as tf

sess = tf.Session()

# 3 + 4 = ?
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

node3 = node1 + node2
print(sess.run(node3))
# Tensor("add:0", shape=(), dtype=float32)
# 7.0
node3 = tf.add(node1, node2)
print(sess.run(node3))
# Tensor("Add_1:0", shape=(), dtype=float32)
# 7.0

sess.close()
