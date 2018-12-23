"""Main DQN agent."""

import numpy as np
import tensorflow as tf
from PIL import Image
import random
from huberLoss import mean_huber_loss, weighted_huber_loss

EPSILON_BEGIN = 1.0
EPSILON_END = 0.1
BETA_BEGIN = 0.5
BETA_END = 1.0

class DQNAgent():
    def __init__(self,
                 eval_model,
                 target_model,
                 memory,
                 num_actions,
                 gamma,
                 update_freq,
                 target_update_freq,
                 update_target_params_ops,
                 batch_size,
                 is_double_dqn,
                 is_per,
                 is_distributional,
                 num_step,
                 is_noisy,
                 learning_rate,
                 rmsp_decay,
                 rmsp_momentum,
                 rmsp_epsilon):

        self._eval_model = eval_model
        self._target_model = target_model
        self._memory = memory
        self._num_actions = num_actions
        self._gamma = gamma
        self._update_freq = update_freq
        self._target_update_freq = target_update_freq
        self._update_target_params_ops = update_target_params_ops
        self._batch_size = batch_size
        self._is_double_dqn = is_double_dqn
        self._is_per = is_per
        self._is_distributional = is_distributional
        self._num_step = num_step
        self._is_noisy = is_noisy
        self._learning_rate = learning_rate
        self._rmsp_decay = rmsp_decay
        self._rmsp_momentum = rmsp_momentum
        self._rmsp_epsilon = rmsp_epsilon
        self._update_times = 0
        self._beta = EPSILON_BEGIN
        self._beta_increment = (EPSILON_END-BETA_BEGIN)/2000000.0
        self._epsilon = EPSILON_BEGIN if is_noisy else 0.
        self._epsilon_increment =  (EPSILON_END - EPSILON_BEGIN)/2000000.0 if is_noisy==0 else 0.
        self._action_ph = tf.placeholder(tf.int32,[None,2],'action_ph')
        self._reward_ph = tf.placeholder(tf.float32,name='reward_ph')
        self._is_terminal_ph = tf.placeholder(tf.float32,name='is_terminal_ph')
        self._action_chosen_by_eval_ph = tf.placeholder(tf.int32,[None,2],'action_chosen_by_eval_ph')
        self._loss_weight_ph = tf.placeholder(tf.float32,name='loss_weight_ph')
        self._error_op,self._train_op = self._get_error_and_train_op(self._reward_ph,self._is_terminal_ph,
                                                                     self._action_ph,self._action_chosen_by_eval_ph,
                                                                     self._loss_weight_ph)


    def _get_error_and_train_op(self,reward_ph,
                                is_terminal_ph,
                                action_ph,
                                action_chosen_by_eval_ph,
                                loss_weight_ph):

        if self._is_distributional == 0:
            q_values_target = self._target_model['q_values']
            q_values_eval = self._eval_model['q_values']

            if self._is_double_dqn:
                max_q = tf.gather_nd(q_values_target,action_chosen_by_eval_ph) # 如果是double-dqn，动作由eval-net选出，q值由target-net得到
            else:
                max_q = tf.reduce_max(q_values_target,axis=1)

            target = reward_ph + (1.0 - is_terminal_ph) * (self._gamma ** self._num_step) * max_q # 这里是多步的dqn
            gathered_outputs = tf.gather_nd(q_values_eval,action_ph,name='gathered_outputs')

            if self._is_per == 1:
                loss = weighted_huber_loss(target,gathered_outputs,loss_weight_ph)
            else:
                loss = mean_huber_loss(target,gathered_outputs)
            train_op = tf.train.RMSPropOptimizer(self._learning_rate,decay=self._rmsp_decay,
                                                 momentum=self._rmsp_momentum,epsilon=self._rmsp_epsilon).minimize(loss)

            error_op = tf.abs(gathered_outputs - target,name='abs_error')
            return train_op,error_op

        else:
            N_atoms = 51
            V_Max = 20.0
            V_Min = 0.0
            Delta_z = (V_Max - V_Min) / (N_atoms - 1)
            z_list = tf.constant([V_Min + i * Delta_z for i in range(N_atoms)], dtype=tf.float32)

            q_distributional_values_target = self._target_model['q_distributional_network'] # batch_size * num_actions * N_atoms
            tmp_batch_size = tf.shape(q_distributional_values_target)[0] # batch_size

            if self._is_double_dqn:
                q_distributional_chosen_by_action_target = tf.gather_nd(q_distributional_values_target,action_chosen_by_eval_ph)
            else:
                action_chosen_by_target_q = tf.cast(tf.argmax(self._target_model['q_values'], axis=1), tf.int32)
                q_distributional_chosen_by_action_target = tf.gather_nd(q_distributional_values_target,
                                                                  tf.concat([tf.reshape(tf.range(tmp_batch_size),[-1,1]),
                                                                             tf.reshape(action_chosen_by_target_q,[-1,1])],axis=1))


            target = tf.tile(tf.reshape(reward_ph,[-1,1]),[1,N_atoms]) + \
                     (self._gamma * self._num_step) * \
                     tf.multiply(tf.reshape(z_list,[1,N_atoms]),(1.0 - tf.tile(tf.reshape(is_terminal_ph,[-1,1]),[1,N_atoms])))

            target = tf.clip_by_value(target,V_Min,V_Max)

            b = (target - V_Min) / Delta_z

            u,l = tf.ceil(b),tf.floor(b)

            u_id,l_id = tf.cast(u,tf.int32),tf.cast(l,tf.int32)

            u_minus_b,b_minus_l = u - b,b - l
            q_distributional_values_eval = self._eval_model['q_distributional_network']

            q_distributional_chosen_by_action_eval = tf.gather_nd(q_distributional_values_eval,action_ph)

            index_help = tf.tile(tf.reshape(tf.range(tmp_batch_size),[-1,1]),[1,N_atoms])

            index_help = tf.expand_dims(index_help,-1) # batch * N_atoms * 1
            u_id = tf.concat([index_help,tf.expand_dims(u_id,-1)],axis=2)
            l_id = tf.concat([index_help,tf.expand_dims(l_id,-1)],axis=2)

            error = q_distributional_chosen_by_action_target * u_minus_b * \
                    tf.log(tf.gather_nd(q_distributional_chosen_by_action_eval, l_id)) \
                    + q_distributional_chosen_by_action_target * b_minus_l * \
                      tf.log(tf.gather_nd(q_distributional_chosen_by_action_eval, u_id))
            error = tf.reduce_sum(error, axis=1)

            if self._is_per == 1:
                loss = tf.negative(error * loss_weight_ph)
            else:
                loss = tf.negative(error)

            train_op = tf.train.RMSPropOptimizer(self._learning_rate,
                                                 decay=self._rmsp_decay, momentum=self._rmsp_momentum,
                                                 epsilon=self._rmsp_epsilon).minimize(loss)
            error_op = tf.abs(error, name='abs_error')
            return error_op, train_op

    def select_action(self,sess,state,epsilon,model):
        batch_size = len(state)
        if np.random.rand() < epsilon:
            action = np.random.randint(0,self._num_actions,size=(batch_size,))
        else:
            state = state.astype(np.float32) / 255.0
            feed_dict = {model['input_frames'] :state}
            action = sess.run(model['action'],feed_dict=feed_dict)
        return action

    def get_multi_step_sample(self,env,sess,num_step,epsilon):
        old_state,action,reward,new_state,is_terminal = env.get_state()
        total_reward = np.sign(reward)
        total_is_terminal = is_terminal

        next_action = self.select_action(sess,new_state,epsilon,self._eval_model)
        env.take_action(next_action)

        for i in range(1,num_step):
            _,_,reward,new_state,is_terminal = env.get_state()
            total_reward += self._gamma ** i * np.sign(reward)
            total_is_terminal += is_terminal
            next_action = self.select_action(sess,new_state,epsilon,self._eval_model)
            env.take_action(next_action)

        return old_state,action,total_reward,new_state,np.sign(total_is_terminal)

    def fit(self,sess,env,num_iterations,do_train=True):

        num_environment = env.num_process
        env.reset()

        for t in range(0,num_iterations,num_environment):
            # 准备数据
            old_state,action,reward,new_state,is_terminal = self.get_multi_step_sample(env,sess,self._num_step,self._epsilon)
            self._memory.append(old_state,action,reward,new_state,is_terminal) # 插入数据
            if self._epsilon > EPSILON_END:
                self._epsilon += num_environment * self._epsilon_increment
            if do_train:
                num_update = sum([1 if i % self._update_freq == 0 else 0 for i in range(t, t + num_environment)])
                # 抽取数据
                for _ in range(num_update):
                    if self._is_per == 1:
                        (old_state_list, action_list, reward_list, new_state_list, is_terminal_list), \
                        idx_list, p_list, sum_p, count = self._memory.sample(self._batch_size)
                    else:
                        old_state_list, action_list, reward_list, new_state_list, is_terminal_list \
                            = self._memory.sample(self._batch_size)

                    feed_dict = {self._target_model['input_frames']: new_state_list.astype(np.float32) / 255.0,
                                 self._eval_model['input_frames']: old_state_list.astype(np.float32) / 255.0,
                                 self._action_ph: list(enumerate(action_list)),
                                 self._reward_ph: np.array(reward_list).astype(np.float32),
                                 self._is_terminal_ph: np.array(is_terminal_list).astype(np.float32),
                                 }

                    if self._is_double_dqn:
                        action_chosen_by_online = sess.run(self._eval_model['action'], feed_dict={
                                    self._eval_model['input_frames']: new_state_list.astype(np.float32)/255.0})
                        feed_dict[self._action_chosen_by_eval_ph] = list(enumerate(action_chosen_by_online))

                    if self._is_per == 1:
                        # Annealing weight beta
                        feed_dict[self._loss_weight_ph] = (np.array(p_list)*count/sum_p)**(-self._beta)
                        error, _ = sess.run([self._error_op, self._train_op], feed_dict=feed_dict)
                        self._memory.update(idx_list, error)
                    else:
                        sess.run(self._train_op, feed_dict=feed_dict)

                    self._update_times += 1
                    if self._beta < BETA_END:
                        self._beta += self._beta_increment

                    if self._update_times%self._target_update_freq == 0:
                        sess.run(self._update_target_params_ops)


    def _get_error(self, sess, old_state, action, reward, new_state, is_terminal):
        '''
        Get TD error for Prioritized Experience Replay
        '''
        feed_dict = {self._target_model['input_frames']: new_state.astype(np.float32)/255.0,
                     self._eval_model['input_frames']: old_state.astype(np.float32)/255.0,
                     self._action_ph: list(enumerate(action)),
                     self._reward_ph: np.array(reward).astype(np.float32),
                     self._is_terminal_ph: np.array(is_terminal).astype(np.float32),
                     }

        if self._is_double_dqn:
            action_chosen_by_online = sess.run(self._eval_model['action'], feed_dict={
                        self._eval_model['input_frames']: new_state.astype(np.float32)/255.0})
            feed_dict[self._action_chosen_by_eval_ph] = list(enumerate(action_chosen_by_online))

        error = sess.run(self._error_op, feed_dict=feed_dict)
        return error

    def get_mean_max_Q(self, sess, samples):
        mean_max = []
        INCREMENT = 1000
        for i in range(0, len(samples), INCREMENT):
            feed_dict = {self._eval_model['input_frames']:
                samples[i: i + INCREMENT].astype(np.float32)/255.0}
            mean_max.append(sess.run(self._eval_model['mean_max_Q'],
                feed_dict = feed_dict))
        return np.mean(mean_max)


    def evaluate(self, sess, env, num_episode):
        """Evaluate num_episode games by online model.
        Parameters
        ----------
        sess: tf.Session
        env: batchEnv.BatchEnvironment
          This is your paralleled Atari environment.
        num_episode: int
          This is the number of episode of games to evaluate
        Returns
        -------
        reward list for each episode
        """
        num_environment = env.num_process
        env.reset()
        reward_of_each_environment  = np.zeros(num_environment)
        rewards_list = []

        num_finished_episode = 0

        while num_finished_episode < num_episode:
            old_state, action, reward, new_state, is_terminal = env.get_state()
            action = self.select_action(sess, new_state, 0, self._eval_model)
            env.take_action(action)
            for i, r, is_t in zip(range(num_environment), reward, is_terminal):
                if not is_t:
                    reward_of_each_environment[i] += r
                else:
                    rewards_list.append(reward_of_each_environment[i])
                    reward_of_each_environment[i] = 0
                    num_finished_episode += 1
        return np.mean(rewards_list), np.std(rewards_list)

