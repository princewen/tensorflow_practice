import numpy as np
import pandas as pd
import time


np.random.seed(3)


N_STATES = 6
ACTIONS = ['left','right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 5   # 最大回合数
FRESH_TIME = 0.3    # 移动间隔时间


def init_q_table(N_STATES,N_ACTIONS):
    table = pd.DataFrame(np.zeros([N_STATES,N_ACTIONS],dtype=np.float32),columns=ACTIONS)
    return table

def choose_action(state,q_table):
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action = np.random.choice(ACTIONS)
    else:
        action = state_actions.argmax()
    return action

def get_env_feedback(state,action):
    if action=='right':
        if state == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = state + 1
            R = 0
    else:
        if state == 0:
            S_ = state
            R = 0
        else:
            S_ = state - 1
            R = 0
    return S_,R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = init_q_table(N_STATES,len(ACTIONS))
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminal = False
        update_env(state, episode, step_counter)
        while not is_terminal:
            action = choose_action(state,q_table)
            S_,R = get_env_feedback(state,action)
            q_predict = q_table.ix[state,action]
            if S_ == 'terminal':
                q_target = R
                is_terminal = True
            else:
                q_target = R + GAMMA * q_table.iloc[S_,:].max()

            q_table.ix[state,action] += ALPHA * (q_target - q_predict)
            state = S_
            update_env(state, episode, step_counter + 1)

            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

