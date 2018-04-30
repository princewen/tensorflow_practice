from maze_env import Maze
from RL_brain import DoubleDQN


def run_maze():
    step = 0
    for episode in range(300):
        observation = env.reset()

        while True:
            env.render()
            action = RL.choose_action(observation)
            observation_,reward,done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = DoubleDQN(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()