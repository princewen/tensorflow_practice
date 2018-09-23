import tensorflow as tf


x = tf.random_normal([1,3])
m = 4
k = 3

d = x.get_shape().as_list()[-1]

W = tf.Variable(tf.random_normal(shape=[d, m, k]))
b = tf.Variable(tf.random_normal(shape = [m, k]))
dot_z = tf.tensordot(x, W, axes=1) + b
z = tf.reduce_max(dot_z, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([x,dot_z,z]))


