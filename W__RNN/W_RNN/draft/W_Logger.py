import numpy as np
import torch

class W_Logger():
    save_path = None
    save_interval = 100
    smooth_interval = 100
    max_episodes = None
    start_episode = None
    is_supervised = False
    supervised_test_interval = 1
    def __init__(self):
        self.start_episode = None
        
    def set(self, save_path= './model', save_interval= 100, smooth_interval = 100):
        self.save_path = save_path
        self.save_interval = save_interval
        self.smooth_interval = smooth_interval

    def setlog(self, info):
        if info is not None:
            print("warning: info overwriten")
        self.info = info

    def init(self, max_episodes):
        nep = max_episodes + 1
        if self.info is not None and self.info['rewards'].shape[0] < nep:
            nep -= self.info['rewards'].shape[0]
        info = dict()
        info['rewards'] = np.zeros(nep)
        info['rewards_smooth'] = np.zeros(nep)
        info['episodelength'] = np.zeros(nep)
        info['episodelength_smooth'] = np.zeros(nep)
        info['rewardrate'] = np.zeros(nep)
        info['rewardrate_smooth'] = np.zeros(nep)
        if self.info is None:
            self.info = info
        else:            
            for i in info.keys():
                self.info[i] = np.hstack((self.info[i], info[i]))
        if hasattr(self, '_init'):
            self._init()

    def update(self, reward, gamelen, gameinfo):
        info = self.info
        episode = self.episode
        info['rewards'][episode] = reward
        info['rewards_smooth'][episode] = info['rewards'][max(0, episode-self.smooth_interval):(episode+1)].mean()
        info['episodelength'][episode] = gamelen
        info['episodelength_smooth'][episode] = info['episodelength'][max(0, episode-self.smooth_interval):(episode+1)].mean()
        info['rewardrate'][episode] = reward/gamelen
        info['rewardrate_smooth'][episode] = info['rewardrate'][max(0, episode-self.smooth_interval):(episode+1)].mean()
        self.info = info
        if hasattr(self, '_update'):
            self._update(gameinfo)
        
    def getdescription(self):
        info = self.info
        reward = info['rewards'][self.episode - 1]
        avR = info['rewards_smooth'][self.episode - 1]
        avL = info['episodelength_smooth'][self.episode - 1]
        avRT = info['rewardrate_smooth'][self.episode - 1]
        if self.is_supervised:
            avR = avR * self.supervised_test_interval
            str = f"Episode {self.episode}/{self.max_episodes}, R {reward:.2f}, avR {avR:.2f}, avERR {avL:.2f}"
        else:
            str = f"Episode {self.episode}/{self.max_episodes}, R {reward:.2f}, avR {avR:.2f}, len {avL:.1f}, rate {avRT:.2f}"
        if hasattr(self, '_getdescription'):
            str = self._getdescription(str)        
        return str
    

        