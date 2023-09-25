import numpy as np
from collections import namedtuple 
import random 
from W_Python.W import W
import torch
import pandas as pd

class W_Buffer:
    memory = []
    def __init__(self, param_buffer, device = None, *arg, **kwarg):
        self.device = device
        param = {"capacity": np.Inf, "mode_sample": "random"}
        param.update(param_buffer)
        self.memory = []
        self.capacity = param_buffer['capacity']
        self.mode_sample = param_buffer['mode_sample']
        self.clear_buffer()

    def clear_buffer(self):
        self.memory = []

    def load(self, file):
        d = pd.read_csv(file)
        colnames = list(d.keys())
        colid = np.where(np.array([x.find('obs') for x in colnames]) == 0)[0]
        obs = d.iloc[:, colid]
        action = d.action.to_numpy()
        # need edits
        pass
    
    def push(self, data):
        data = W.W_enlist(data)
        self.memory += data
        while len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, n = None, mode_sample = None):
        if mode_sample is None:
            mode_sample = self.mode_sample
        if mode_sample == "last":
            d = self.sample_last(n)
        elif mode_sample == "random":
            d = self.sample_random(n)
        elif mode_sample == "all":
            d = self.memory
            
        keys = list(d[0].keys())
        tuple_buffer = namedtuple('Buffer', keys)
        
        buffer = []
        for i in keys:
            buffer += [torch.concat([x[i].unsqueeze(0) for x in d])]
        
        out = tuple_buffer(*buffer)
        return out
    
    def sample_last(self, n = 1):
        memory = self.memory[-n:]
        assert len(memory) == n
        return memory
    
    def sample_random(self, n = 1):
        memory = random.sample(self.memory, n)
        return memory