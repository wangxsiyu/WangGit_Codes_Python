import torch
from collections import namedtuple 
import numpy as np
import random

class W_Buffer:
    def __init__(self, tuple, capacity = np.Inf, mode_sample = "random", *arg, **kwarg):
        self.memory = []
        self.tuple = tuple
        self.capacity = capacity
        self.mode_sample = mode_sample
        self.clear()

    def clear(self):
        self.buffer = []

    def add(self, *arg):
        d = [np.array(x, dtype = 'float') for x in arg]
        d = self.tuple(*d)
        self.buffer += [d]
        
    def push(self):
        buffer = self.tuple(*zip(*self.buffer))
        buffer = [np.concatenate(x) for x in buffer]
        buffer = self.tuple(*buffer)
        self.memory += [buffer]
        while len(self.memory) > self.capacity:
            self.memory.pop(0)
        self.buffer = []

    def get_last(self, n = 1):
        return self.memory[-n:]
    
    def get_random(self, n = 1):
        sub = random.sample(self.memory, n)
        return sub

    def sample(self, n):
        if self.mode_sample == "last":
            d = self.get_last(n)
        elif self.mode_sample == "random":
            d = self.get_random(n)
        d = self.tuple(*zip(*d))
        d = [np.stack(x) for x in d]
        d = [torch.from_numpy(x).float() for x in d]
        d = self.tuple(*d)
        return d

class W_Worker:
    def __init__(self, env, model, *arg, **kwarg):
        self.env = env
        self.model = model
        Memory = namedtuple('Memory', ('obs', 'action', 'reward', 'timestep', 'done'))
        self.memory = W_Buffer(Memory, *arg, **kwarg)
        # self.set_mode(*arg, **kwarg)

    def set_mode(self, mode_worker = "Test"):
        self.mode_worker = mode_worker

    def select_action(self, action_dist, mode_action):
        if mode_action == "softmax":
            action_cat = torch.distributions.Categorical(action_dist.squeeze())
            action = action_cat.sample()
        return action

    def run_episode(self, mode_action = "softmax"):
        self.model.eval()
        done = False
        total_reward = 0
        obs = self.env.reset()
        mem_state = None
        self.memory.clear()
        while not done:
            # take actions
            obs = torch.from_numpy(obs).unsqueeze(0).float()
            action_dist, val_estimate, mem_state_new = self.model(obs.unsqueeze(0), mem_state)
            action = self.select_action(action_dist, mode_action)

            obs_new, reward, done, timestep, _ = self.env.step(action)
            reward = float(reward)
            action_onehot = torch.nn.functional.one_hot(action, self.env.action_space.n)
            action_onehot = action_onehot.unsqueeze(0).float()
            
            self.memory.add(obs, action_onehot, [reward], [timestep], [done])
            
            obs = obs_new
            mem_state = mem_state_new
            total_reward += reward

        self.memory.push()
        return total_reward

    def run_episode_outputlayer(self, buffer):
        self.model.train()
        (obs, action, reward, timestep, done) = buffer
        mem_state = None
        action_dist, val_estimate, mem_state = self.model(obs, mem_state)
        action_dist = action_dist.permute((1,0,2))
        eps = 1e-5
        action_dist = action_dist.clamp(eps, 1-eps)
        val_estimate = val_estimate.permute((1,0,2)).squeeze(2)
        action_likelihood = (action_dist * action).sum(-1)
        tb = namedtuple('TrainingBuffer', ("action_dist","value", "action_likelihood"))
        return tb(action_dist, val_estimate, action_likelihood)

    def run_worker(self, nrep = 1):
        rs = []
        for _ in range(nrep):
            r = self.run_episode()
            rs.append(r)
        return np.mean(rs)
        

