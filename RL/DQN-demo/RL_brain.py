import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)


class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter = 300,
                 memory_size = 500,
                 batch_size = 32,
                 e_greedy_increment=None,
                 output_graph=False):


        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        # features 是坐标，记忆(s,a,r,s')
        self.memory = np.zeros((self.memory_size,n_features * 2 + 2))

        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter('logs/',self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.cost_his = []

    def _build_net(self):
        #-----------------inputs------------------#
        self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s')
        self.s_ = tf.placeholder(tf.float32,[None,self.n_features],name='s_')
        self.r = tf.placeholder(tf.float32,[None,],name='r')
        self.a = tf.placeholder(tf.int32,[None,],name='a')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s,20,tf.nn.relu,kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer,name='e1'
                                 )

            self.q_eval = tf.layers.dense(e1,self.n_actions,kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer,name='q')

        # ------------------ build target_net ------------------

        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')


        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next,axis=1,name='Qmax_s_')
            # 一个节点被 stop之后，这个节点上的梯度，就无法再向前BP了
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            # tf.stack
            #a = tf.constant([1,2,3])
            # b = tf.constant([4,5,6])
            # c = tf.stack([a,b],axis=1)
            # [[1 4]
            #  [2 5]
            # [3 6]]
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            # 用indices从张量params得到新张量
            # indices = [[0, 0], [1, 1]]
            # params = [['a', 'b'], ['c', 'd']]
            # output = ['a', 'd']
            # 这里self.q_eval是batch * action_number,a_indices是batch * 1，也就是说选择当前估计每个动作的Q值
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval_wrt_a,name='TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


    def store_transition(self,s,a,r,s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # hstack:Stack arrays in sequence horizontally
        transition = np.hstack((s,[a,r],s_))
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1


    def choose_action(self,observation):
        observation = observation[np.newaxis,:]
        if np.random.random() < self.epsilon:
            action_values = self.sess.run(self.q_eval,feed_dict={self.s:observation})
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0,self.n_actions)

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size = self.batch_size)

        batch_memory = self.memory[sample_index,:]

        _,cost = self.sess.run(
            [self._train_op,self.loss],
            feed_dict={
                self.s:batch_memory[:,:self.n_features],
                self.a:batch_memory[:,self.n_features],
                self.r:batch_memory[:,self.n_features+1],
                self.s_:batch_memory[:,-self.n_features:]
            }
        )

        self.cost_his.append(cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()













