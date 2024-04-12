import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1 data
x_train = np.array([[[[1],[2],[3]],
                     [[4],[5],[6]],
                     [[7],[8],[9]]]])
# print(x_train.shape)    # (1, 3, 3, 1)

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 3, 3, 1])
w = tf.compat.v1.constant([[[[1.]], [[0.]]],
                           [[[1.]], [[0.]]]])
# print(w)    # Tensor("Const:0", shape=(2, 2, 1, 1), dtype=float32)
#             # 2,2 커널 / 1 채널 / 1 아웃풋 필터

L1 = tf.nn.conv2d(x, w, strides=(1,1,1,1), padding='VALID')
# print(L1) # (?, 2, 2, 1)

sess = tf.compat.v1.Session()
output = sess.run(L1, feed_dict={x:x_train})
print("="*20, "결과", "="*20)
print(output)
print("="*20, "결과", "="*20)



sess.close()



