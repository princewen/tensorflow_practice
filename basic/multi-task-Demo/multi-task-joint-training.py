import tensorflow as tf

import numpy as np

x = tf.placeholder("float",[None,10],name='X')
y1 = tf.placeholder("float",[None,1],name='Y1')
y2 = tf.placeholder("float",[None,1],name='Y2')

shared_layer_weights = tf.get_variable(name='share_W',shape=[10,20],initializer=tf.random_normal_initializer(0.0,0.1))
y1_layer_weights = tf.get_variable(name='share_Y1',shape=[20,1],initializer=tf.random_normal_initializer(0.0,0.1))
y2_layer_weights = tf.get_variable(name='share_Y2',shape=[20,1],initializer=tf.random_normal_initializer(0.0,0.1))

shared_layer = tf.nn.relu(tf.matmul(x , shared_layer_weights))
y1_layer = tf.nn.relu(tf.matmul(shared_layer,y1_layer_weights))
y2_layer = tf.nn.relu(tf.matmul(shared_layer,y2_layer_weights))

y1_loss = tf.reduce_sum(tf.squared_difference(y1,y1_layer))
y2_loss = tf.reduce_sum(tf.squared_difference(y2,y2_layer))
joint_loss = y1_loss + y2_loss

train_op = tf.train.AdamOptimizer(0.1).minimize(joint_loss)


print(np.random.rand(10,1).tolist())


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iter in range(10):
        _,loss = sess.run([train_op,joint_loss],feed_dict={
        x: np.random.rand(10, 10).tolist(),
        y1: np.random.rand(10,1).tolist(),
        y2: np.random.rand(10, 1).tolist()
        })
        print(loss)
