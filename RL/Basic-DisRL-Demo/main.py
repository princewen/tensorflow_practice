
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import numpy as np
import pickle
from Config import Categorical_DQN_Config
from Categorical_DQN import Categorical_DQN


def map_scores(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()

def BreakOut_CDQN(index, env):
    with tf.variable_scope('DQfD_' + str(index)):
        agent = Categorical_DQN(env, Categorical_DQN_Config())
    scores = []
    for e in range(Categorical_DQN_Config.episode):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        # while done is False:
        last_lives = 5
        throw = True
        items_buffer = []
        while not done:
            env.render()
            action = 1 if throw else agent.greedy_action(state)
            next_state, real_reward, done, info = env.step(action)
            lives = info['ale.lives']
            train_reward = 1 if throw else -1 if lives < last_lives else real_reward
            score += real_reward
            throw = lives < last_lives
            last_lives = lives
            # agent.train(state, train_reward, [action], next_state, 0.1)
            items_buffer.append([state, [action], next_state, 0.1])
            state = next_state
            if train_reward != 0:  # train when miss the ball or score or throw the ball in the beginning
                for item in items_buffer:
                    agent.train(item[0], -1 if throw else train_reward, item[1], item[2], item[3])
                items_buffer = []
        scores.append(score)
        agent.save_model()
        # if np.mean(scores[-min(10, len(scores)):]) > 495:
        #     break
    return scores


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    CDQN_sum_scores = np.zeros(Categorical_DQN_Config.episode)
    for i in range(Categorical_DQN_Config.iteration):
        scores = BreakOut_CDQN(i,env)
        c51_sum_scores = [a + b for a, b in zip(scores, CDQN_sum_scores)]
    C51DQN_mean_scores = CDQN_sum_scores / Categorical_DQN_Config.iteration
    with open('/Users/mahailong/C51DQN/C51DQN_mean_scores.p', 'wb') as f:
        pickle.dump(C51DQN_mean_scores, f, protocol=2)