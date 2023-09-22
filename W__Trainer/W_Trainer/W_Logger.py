import torch
import numpy as np
from W_Python.W import W
import os

class W_Logger():
    metadata_logger = {"save_interval": 1000}
    def __init__(self, param_logger):
        self.metadata_logger.update(param_logger)
        self.last_saved_filename = None
        self.info = []

    def initialize(self, max_episodes, start_episode = 0):
        self.max_episodes = max_episodes
        self.episode = start_episode
        tqdmrange = range(start_episode, self.max_episodes)
        return tqdmrange

    def update0(self, reward, info_loss, newdata):
        if len(self.info) == 0:
            info = dict(reward = reward)
            self.info += [info]

    def update(self, reward, info_loss, newdata):
        self.episode += 1
        info = dict(reward = reward)
        self.info += [info]

    def getdescription(self):
        rs = [x['reward'] for x in self.info]
        smoothstart = max(0, len(rs) - 100)
        avr = np.mean(rs[smoothstart:])
        str = f"avR:{avr:.2f}"
        return str

    def save(self, savepath, state_dict):
        if savepath is not None:
            if (self.episode) % self.metadata_logger['save_interval'] == 0:
                save_name = "iteration_{epi:09d}".format(epi=self.episode) + ".pt"
                save_path = os.path.join(savepath, save_name)
                self.last_saved_filename = save_path
                torch.save({
                    "state_dict": state_dict,
                    "training_info": self.info,
                }, save_path)  

    def setlog(self, info = None):
        if info is None:
            info = []
        if self.info is not None:
            print("warning: overwriting existing info")
        self.info = info

