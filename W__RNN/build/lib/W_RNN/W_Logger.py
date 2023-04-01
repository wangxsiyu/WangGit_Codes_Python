import numpy as np
import torch

class W_Logger():
    save_path = None
    save_interval = 100
    smooth_interval = 100
    last_saved_version = None
    max_episodes = None
    info = None
    def __init__(self):
        pass
        
    def set(self, save_path= './model', save_interval= 100, smooth_interval = 100):
        self.save_path = save_path
        self.save_interval = save_interval
        self.smooth_interval = smooth_interval

    def setlog(self, info):
        if info is not None:
            print("warning: info overwriten")
        self.info = info

    def init(self, max_episodes):
        self.episode = 0
        self.max_episodes = max_episodes
        max_episodes += 1
        info = dict()
        info['rewards'] = np.zeros(max_episodes)
        info['rewards_smooth'] = np.zeros(max_episodes)
        info['episodelength'] = np.zeros(max_episodes)
        info['episodelength_smooth'] = np.zeros(max_episodes)
        info['rewardrate'] = np.zeros(max_episodes)
        info['rewardrate_smooth'] = np.zeros(max_episodes)
        if self.info is None:
            self.info = info
        else:
            for i in info.keys():
                self.info[i] = np.hstack((self.info[i], info[i]))

    def update(self, reward, gamelen):
        info = self.info
        episode = self.episode
        info['rewards'][episode] = reward
        info['rewards_smooth'][episode] = info['rewards'][max(0, episode-self.smooth_interval):(episode+1)].mean()
        info['episodelength'][episode] = gamelen
        info['episodelength_smooth'][episode] = info['episodelength'][max(0, episode-self.smooth_interval):(episode+1)].mean()
        info['rewardrate'][episode] = reward/gamelen
        info['rewardrate_smooth'][episode] = info['rewardrate'][max(0, episode-self.smooth_interval):(episode+1)].mean()
        self.episode += 1
        self.info = info

    def getdescription(self):
        info = self.info
        reward = info['rewards'][self.episode - 1]
        avR = info['rewards_smooth'][self.episode - 1]
        avL = info['episodelength_smooth'][self.episode - 1]
        avRT = info['rewardrate_smooth'][self.episode - 1]
        str = f"Episode {self.episode}/{self.max_episodes}, R {reward:.2f}, avR {avR:.2f}, len {avL:.1f}, rate {avRT:.2f}"
        return str
    
    def save(self, state_dict):
        if self.save_path is not None:
            save_path = self.save_path + "_{epi:09d}".format(epi=self.episode) + ".pt"
            self.last_saved_version = save_path
            if (self.episode) % self.save_interval == 0:
                torch.save({
                    "state_dict": state_dict,
                    "training_info": self.info,
                }, save_path)  

        