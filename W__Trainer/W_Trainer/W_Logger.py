import torch
import numpy as np
from W_Python.W import W
import os

class W_Logger():
    info0 = {'info':[], 'history':[]}
    metadata_logger = {"save_interval": 1000, 'output_smooth': 100, 'supervised_test_interval': None}
    def __init__(self, param_logger):
        self.metadata_logger.update(param_logger)
        self.last_saved_filename = None
        self.info = self.info0.copy()

    def initialize(self, max_episodes, start_episode = 0, info = None):
        if info is not None:
            if start_episode == 0:
                self.info['history'] += [info]
                self.info['info'] = []
            else:
                if self.info != self.info0:
                    print("warning: overwriting existing info")
                self.info = info
        self.max_episodes = max_episodes
        self.episode = start_episode
        tqdmrange = range(start_episode, self.max_episodes)
        return tqdmrange

    def update0(self, *arg, **kwarg):
        if len(self.info['info']) == 0:
            self.update(*arg, **kwarg)

    def update1(self, *arg, **kwarg):
        self.episode += 1
        self.update(*arg, **kwarg)

    def update(self, reward, info_loss, newdata):
        info = dict(reward = reward)
        self.info['info'] += [info]

    def getdescription(self):
        rs = [x['reward'] for x in self.info['info']]
        smoothstart = max(0, len(rs) - self.metadata_logger['output_smooth'])
        avr = np.nanmean(rs[smoothstart:])
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
