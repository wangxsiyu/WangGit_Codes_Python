import torch
from collections import namedtuple 

Data_Episode_Tuple_Train = namedtuple('Data_Episode_Tuple_Train', ('obs', 'action', 'reward', 'timestep', 'done', 'action_prob', 'value'))
Data_Episode_Tuple_Test = namedtuple('Data_Episode_Tuple_Test', ('obs', 'action', 'reward', 'timestep', 'done'))
class W_Buffer:
    def __init__(self, tuple):
        self.memory = []
        self.tuple = tuple
        self.clear()

    def clear(self):
        self.buffer = []

    def add(self, *arg):
        self.buffer += [self.tuple(*arg)]

    def push(self):
        self.memory += [self.buffer]
        self.buffer = []

    def get_last(self):
        return self.memory[-1]

class W_Worker:
    def __init__(self, env, model, *arg, **kwarg):
        self.env = env
        self.model = model
        self.set_mode(*arg, **kwarg)

    def set_mode(self, mode = "test", memory = None):
        self.worker_mode = mode
        if memory is None:
            if mode == "test":
                self.memory = W_Buffer(Data_Episode_Tuple_Test)
            elif mode == "train":
                self.memory = W_Buffer(Data_Episode_Tuple_Train)
            elif mode == "record":
                pass
                # self.memory = W_Buffer(('obs', 'action', 'reward', 'timestep', 'done', 'electrode', "recording"))    
        else:
            self.memory = memory
    
    def run_episode(self):
        done = False
        total_reward = 0
        obs = self.env.reset()
        mem_state = None
        self.memory.clear()
        while not done:
            # take actions
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).float()
            action_dist, val_estimate, mem_state = self.model(obs, mem_state)
            action_cat = torch.distributions.Categorical(action_dist.squeeze())
            action = action_cat.sample()

            obs_new, reward, done, timestep, _ = self.env.step(action)
            action_onehot = torch.nn.functional.one_hot(action, self.env.action_space.n)
            action_onehot = action_onehot.unsqueeze(0).float()
            if self.worker_mode == "test":
                self.memory.add(obs, action_onehot, reward, timestep, done)
            elif self.worker_mode == "train":
                self.memory.add(obs, action_onehot, reward, timestep, done, action_dist, val_estimate)
            elif self.worker_mode == "record":
                pass

            obs = obs_new
            total_reward += reward

        self.memory.push()
        return total_reward


            

