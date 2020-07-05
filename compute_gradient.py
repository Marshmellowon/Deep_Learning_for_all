# gradient custom
import os
# compat.v1을 사용
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X = [1, 2, 3]
Y = [1, 2, 3]
W = tf.Variable(5.)

hypothesis = X * W
gradient = tf.reduce_mean((W * X - Y) * X) * 2
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optmizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# get gradient
gvs = optmizer.compute_gradients(cost)
# apply gradient
apply_gradient = optmizer.apply_gradients(gvs)

# launch the graph in a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradient)
