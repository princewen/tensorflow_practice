import tensorflow as tf

class Discriminator:
    def __init__(self, env):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """

        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=env.action_space.n)
            # add noise for stabilise training
            expert_a_one_hot += tf.random_normal(tf.shape(expert_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1) # 将专家的state和action进行拼接

            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])
            agent_a_one_hot = tf.one_hot(self.agent_a, depth=env.action_space.n)
            # add noise for stabilise training
            agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1) # 将agent的state和action进行拼接

            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(input=expert_s_a)
                network_scope.reuse_variables()  # share parameter
                prob_2 = self.construct_network(input=agent_s_a)

            with tf.variable_scope('loss'): # 其实就是对数损失函数，想把专家行为与agent行为进行区分
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
                loss = loss_expert + loss_agent
                loss = -loss
                tf.summary.scalar('discriminator', loss)

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss)

            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent

    def construct_network(self, input):
        """
        得到采取该动作的概率，对于专家的行动来说，D希望越大越好，对于agent的行动来说，D希望越小越好
        :param input:
        :return:
        """
        layer_1 = tf.layers.dense(inputs=input, units=20, activation=tf.nn.leaky_relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name='layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=20, activation=tf.nn.leaky_relu, name='layer3')
        prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.sigmoid, name='prob')
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, agent_s, agent_a):
        """
        返回agent得到的奖励，对于agent来说，希望D输出的概率越大越好
        :param agent_s:
        :param agent_a:
        :return:
        """
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

