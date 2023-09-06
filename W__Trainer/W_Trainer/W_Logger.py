import torch

class W_Logger():
    metadata_logger = {"save_path": None, "save_interval": 1000}
    info = None
    def __init__(self, param_logger):
        self.metadata_logger.update(param_logger)
        self.last_saved_filename = None
        self.start_episode = None

    def initialize(self, max_episodes):
        self.episode =  self.get_start_episode()
        self.max_episodes = max_episodes
        tqdmrange = range(self.episode, self.max_episodes)
        return tqdmrange
    
    def get_start_episode(self):
        if self.start_episode is None:
            self.start_episode = 0
        return self.start_episode

    def update(self, reward, info_loss, newdata):
        self.episode += 1

    def save(self, state_dict):
        if self.save_path is not None:
            if (self.episode) % self.metadata_logger['save_interval'] == 0:
                save_path = self.save_path + "_{epi:09d}".format(epi=self.episode) + ".pt"
                self.last_saved_filename = save_path
                torch.save({
                    "state_dict": state_dict,
                    "training_info": self.info,
                }, save_path)  



        # if self.logger.get_start_episode() >= max_episodes:
        #     print(f'model already trained: total steps = {max_episodes}, skip')
        #     return

        # tqdmrange = range(self.logger.get_start_episode()+1, max_episodes+1)