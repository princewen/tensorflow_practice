import tensorflow as tf
import numpy as np
import random
from collections import deque

from utils import conv,noisy_dense

class NoisyNetDQN():
    def __init__(self,env,config):
        self.sess = tf.InteractiveSession()
        self.config = config

        self.replay_buffer = deque(maxlen = self.config.replay_buffer_size)
        self.time_step = 0

        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

        print('state_dim:', self.state_dim)
        print('action_dim:', self.action_dim)

        self.action_batch = tf.placeholder('int32',[None])
        self.y_input = tf.placeholder('float',[None,self.action_dim])

        batch_shape = [None]
        batch_shape.extend(self.state_dim)

        self.eval_input = tf.placeholder('float',batch_shape)
        self.target_input = tf.placeholder('float',batch_shape)

        self.build_noisy_dqn_net()

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        self.save_model()
        self.restore_model()

    def build_layers(self,state,c_names,units_1,units_2,w_i,b_i,reg=None):
        with tf.variable_scope('conv1'):
            conv1 = conv(state,[5,5,3,6],[6],[1,2,2,1],w_i,b_i)
        with tf.variable_scope('conv2'):
            conv2 = conv(conv1,[3,3,6,12],[12],[1,2,2,1],w_i,b_i)
        with tf.variable_scope('flatten'):
            flatten = tf.contrib.layers.flatten(conv2)

        with tf.variable_scope('dense1'):
            dense1 = noisy_dense(flatten,units_1,[units_1],c_names,w_i,b_i,noisy_distribution = self.config.noisy_distribution)

        with tf.variable_scope('dense2'):
            dense2 = noisy_dense(dense1,units_2,[units_2],c_names,w_i,b_i,noisy_distribution = self.config.noisy_distribution)

        with tf.variable_scope('dense3'):
            dense3 = noisy_dense(dense2,self.action_dim,[self.action_dim],c_names,w_i,b_i,noisy_distribution = self.config.noisy_distribution)

        return dense3

    def build_noisy_dqn_net(self):
        with tf.variable_scope('target_net'):
            c_names = ['target_net_arams',tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1,0.1)
            b_i = tf.constant_initializer(0.1)
            self.q_target = self.build_layers(self.target_input,c_names,24,24,w_i,b_i)

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1,0.1)
            b_i = tf.constant_initializer(0.1)
            self.q_eval = self.build_layers(self.eval_input,c_names,24,24,w_i,b_i)

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval,self.y_input))

        self.optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE).minimize(self.loss)

        eval_params = tf.get_collection("eval_net_params")
        target_params = tf.get_collection('target_net_params')

        self.update_target_net = [tf.assign(t,e) for t,e in zip(target_params,eval_params)]


    def save_model(self):
        print("Model saved in : ", self.saver.save(self.sess, self.config.MODEL_PATH))

    def restore_model(self):
        self.saver.restore(self.sess, self.config.MODEL_PATH)
        print("Model restored.")


    def perceive(self,state,action,reward,next_state,done):
        self.replay_buffer.append((state,action,reward,next_state,done))


    def train_q_network(self,update=True):

        if len(self.replay_buffer) < self.config.START_TRAINING:
            return

        self.time_step += 1
        minibatch = random.sample(self.replay_buffer,self.config.BATCH_SIZE)

        np.random.shuffle(minibatch)

        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done = [data[4] for data in minibatch]

        q_target = self.sess.run(self.q_target,feed_dict={self.target_input:next_state_batch})
        q_eval = self.sess.run(self.q_eval,feed_dict={self.eval_input:state_batch})

        done = np.array(done) + 0

        # DQN的结构 r + max q_target[a]
        y_batch = np.zeros((self.config.BATCH_SIZE,self.action_dim))
        for i in range(0,self.config.BATCH_SIZE):
            temp = q_eval[i]
            action = np.argmax(q_target[i])
            temp[action_batch[i]] = reward_batch[i] + (1 - done[i]) * self.config.GAMMA * q_target[i][action]
            y_batch[i] = temp


        self.sess.run(self.optimizer,feed_dict={
            self.y_input:y_batch,
            self.eval_input:state_batch,
            self.action_batch:action_batch
        })

        if update and self.time_step % self.config.UPDATE_TARGET_NET == 0:
            self.sess.run(self.update_target_net)



    def noisy_action(self, state):

        return np.argmax(self.sess.run(self.q_target,feed_dict={self.target_input: [state]})[0])











