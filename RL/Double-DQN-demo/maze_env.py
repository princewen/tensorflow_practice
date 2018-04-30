import numpy as np
import time
import sys
import tkinter as tk


UNIT = 50
MAZE_H = 4
MAZE_W = 4

class Maze(tk.Tk,object):
    def __init__(self):
        super(Maze,self).__init__()
        self.action_space = ['u','d','l','r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry("{}x{}".format(MAZE_H * UNIT,MAZE_W * UNIT))
        self._build_maze()


    def _build_maze(self):
        self.canvas = tk.Canvas(self,bg='white',
                                height = MAZE_H * UNIT,
                                width = MAZE_W * UNIT)


        # create grids
        for c in range(0,MAZE_W*UNIT,UNIT):
            x0,y0,x1,y1 = c,0,c,MAZE_H * UNIT
            self.canvas.create_line(x0,y0,x1,y1)

        for r in range(0,MAZE_H*UNIT,UNIT):
            x0,y0,x1,y1 = 0,r,MAZE_W * UNIT,r
            self.canvas.create_line(x0,y0,x1,y1)

        origin = np.array([25,25])
        hell1_center = origin + np.array([UNIT * 2,UNIT])

        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 20,hell1_center[1] - 20,
            hell1_center[0] + 20,hell1_center[1] + 20,
            fill = 'black'
        )

        oval_center = origin + UNIT * 2

        self.oval = self.canvas.create_oval(
            oval_center[0] - 20,oval_center[1] - 20,
            oval_center[0] + 20,oval_center[1] + 20,
            fill = 'red'
        )

        self.rect = self.canvas.create_rectangle(
            origin[0] - 20,origin[1] - 20,
            origin[0] + 20,origin[1] + 20
        )

        self.canvas.pack()


    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)

        origin = np.array([25,25])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 20,origin[1] - 20,
            origin[0] + 20,origin[1] + 20,
            fill = 'yellow'
        )
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)



    def step(self,action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0,0])

        if action == 0:
            if s[1] > UNIT:
                base_action[1] -= UNIT

        elif action == 1:
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT

        elif action == 2:
            if s[0] < (MAZE_W - 1 ) *UNIT:
                base_action[0] += UNIT

        elif action == 3:
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect,base_action[0],base_action[1])

        next_coords = self.canvas.coords(self.rect)

        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True

        elif next_coords in [self.canvas.coords(self.hell1)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()


