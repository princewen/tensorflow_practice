import tensorflow as tf
import numpy as np

class Critic(object):

    def __init__(self,sess,n_features,gamma = 0.9,lr=0.01):

        self.sess = sess
        self.s = tf.placeholder(tf.float32,[1,n_features],name='state')
        self.v_ = tf.placeholder(tf.float32,[1,1],name='v_next')
        self.r = tf.placeholder(tf.float32,None,name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(0,0.1),
                bias_initializer = tf.constant_initializer(0.1),
                name = 'l1'
            )

            self.v = tf.layers.dense(
                inputs = l1,
                units = 1,
                activation = None,
                kernel_initializer=tf.random_normal_initializer(0,0.1),
                bias_initializer = tf.constant_initializer(0.1),
                name = 'V'
            )


        with tf.variable_scope('squared_TD_error'):
            self.td_error  = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)


        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)


    def learn(self,s,r,s_):
        s,s_ = s[np.newaxis,:],s_[np.newaxis,:]

        v_ = self.sess.run(self.v,feed_dict = {self.s:s_})

        td_error,_ = self.sess.run([self.td_error,self.train_op],
                                   feed_dict={self.s:s,self.v_:v_,self.r:r})

        return td_error



