# 당뇨병 예측
import os
# compat.v1을 사용
import tensorflow.compat.v1 as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# get csv data
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# X, Y, W, b
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([8, 1]), name='weight')  # 가중치
b = tf.Variable(tf.random_normal([1]), name='bias')  # 편향

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)  # sigmoid 사용 matmul이 뭔지 ??

# cost / loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False  casting
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Train the model
# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: \n", h, "\nCorrect (Y): \n", c, "\nAccuracy: ", a)

# TODO: tf.decode_csv?? 해보기
# TODO: 캐글에서 classification으로 예측 해보기
