import numpy as np
import tensorflow as tf

import make_env

from model_agent_maddpg import MADDPG
from replay_buffer import ReplayBuffer

gpu_fraction = 0.4


def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in
                     zip(online_var, target_var)]

    return target_init, target_update


agent1_ddpg = MADDPG('agent1')
agent1_ddpg_target = MADDPG('agent1_target')

agent2_ddpg = MADDPG('agent2')
agent2_ddpg_target = MADDPG('agent2_target')

agent3_ddpg = MADDPG('agent3')
agent3_ddpg_target = MADDPG('agent3_target')

saver = tf.train.Saver()

agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')
agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')

agent2_actor_target_init, agent2_actor_target_update = create_init_update('agent2_actor', 'agent2_target_actor')
agent2_critic_target_init, agent2_critic_target_update = create_init_update('agent2_critic', 'agent2_target_critic')

agent3_actor_target_init, agent3_actor_target_update = create_init_update('agent3_actor', 'agent3_target_actor')
agent3_critic_target_init, agent3_critic_target_update = create_init_update('agent3_critic', 'agent3_target_critic')


def get_agents_action(o_n, sess, noise_rate=0):
    agent1_action = agent1_ddpg.action(state=[o_n[0]], sess=sess) + np.random.randn(2) * noise_rate
    agent2_action = agent2_ddpg.action(state=[o_n[1]], sess=sess) + np.random.randn(2) * noise_rate
    agent3_action = agent3_ddpg.action(state=[o_n[2]], sess=sess) + np.random.randn(2) * noise_rate
    return agent1_action, agent2_action, agent3_action


if __name__ == '__main__':
    env = make_env.make_env('simple_tag')
    o_n = env.reset()

    agent_reward_v = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_reward_op = [tf.summary.scalar('agent' + str(i) + '_reward', agent_reward_v[i]) for i in range(3)]

    agent_a1 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_a1_op = [tf.summary.scalar('agent' + str(i) + '_action_1', agent_a1[i]) for i in range(3)]

    agent_a2 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_a2_op = [tf.summary.scalar('agent' + str(i) + '_action_2', agent_a2[i]) for i in range(3)]

    reward_100 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    reward_100_op = [tf.summary.scalar('agent' + str(i) + '_reward_l100_mean', reward_100[i]) for i in range(3)]

    reward_1000 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    reward_1000_op = [tf.summary.scalar('agent' + str(i) + '_reward_l1000_mean', reward_1000[i]) for i in range(3)]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

    sess = tf.Session(config=config)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init,
              agent2_actor_target_init, agent2_critic_target_init,
              agent3_actor_target_init, agent3_critic_target_init])

    saver.restore(sess, './three_ma_weight/6000.cptk')
    summary_writer = tf.summary.FileWriter('./three_ma_summary', graph=tf.get_default_graph())

    e = 1

    reward_100_list = [[], [], []]
    for i in range(1000000):
        env.render()
        if i % 1000 == 0:
            o_n = env.reset()
            for agent_index in range(3):
                summary_writer.add_summary(sess.run(reward_1000_op[agent_index],
                                                    {reward_1000[agent_index]: np.mean(reward_100_list[agent_index])}),
                                           i // 1000)

        agent1_action, agent2_action, agent3_action = get_agents_action(o_n, sess, noise_rate=0.2)

        a = [[0, i[0][0], 0, i[0][1], 0] for i in [agent1_action, agent2_action, agent3_action]]

        a.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

        o_n_next, r_n, d_n, i_n = env.step(a)

        o_n = o_n_next

    sess.close()
