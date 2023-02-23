import W_Python as W
import gym
import numpy as np



class W_Gym(W_Gym_render):

class W_Gym_render(W_Gym_base):
    currentscreen = None

    def __init__(self, render_mode = None, *arg):
        super().__init__(*arg)

    def flip(self, is_clear = True):
        self.obs = self.currentscreen
        if is_clear:
            self.blankscreen()

    def blankscreen(self):
        self.currentscreen = np.zeros(self.observation_space.shape)

class W_Gym_base(gym.Env):
    R_default = {"R_advance":0, "R_error": -1, "R_reward": 1}
    def __init__(self, dt = 1):
        

    def get_probabilistic_reward(self, p):
        if np.random.uniform() <= p:
            return self.R_default['R_reward']
        else:
            return 0
