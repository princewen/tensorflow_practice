from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(10):
        observation = env.reset()
        while True:
            env.render()
            action = RL.choose_action(str(observation))

            observation_,reward,done = env.step(action)

            RL.learn(str(observation),action,reward,str(observation_))

            observation = observation_

            if done:
                break

    print('game over')
    print(RL.q_table)
    env.destroy()





if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # tkinter 在100ms后调用 after
    env.after(100,update)
    env.mainloop()