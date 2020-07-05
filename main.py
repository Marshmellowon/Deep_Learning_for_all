import os
# compat.v1을 사용
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add = a + b

print(sess.run(add, feed_dict={a: 3, b: 4.5}))
print(sess.run(add, feed_dict={a: [1, 3], b: [2, 4]}))
