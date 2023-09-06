import numpy as np
from collections import namedtuple 
import random 

class W_Buffer:
    buffer = []
    memory = []
    def __init__(self, param_buffer, device = None, *arg, **kwarg):
        self.device = device
        param = {"capacity": np.Inf, "mode_sample": "random", "tuple": ("obs","action","reward","isdone")}
        param.update(param_buffer)
        self.tuple_buffer = namedtuple('Buffer', param['tuple'])
        self.memory = []
        self.capacity = param_buffer['capacity']
        self.mode_sample = param_buffer['mode_sample']
        self.clear_buffer()

    def clear_buffer(self):
        self.buffer = []

    def add_to_buffer(self, *arg):
        d = [np.array(x, dtype = 'float') for x in arg]
        d = self.tuple_buffer(*d)
        self.buffer += [d]
    
    def push(self):
        # buffer = self.tuple(*zip(*self.buffer))
        # buffer = [np.concatenate(x) for x in buffer]
        # buffer = self.tuple(*buffer)
        self.memory += [self.buffer]
        while len(self.memory) > self.capacity:
            self.memory.pop(0)
        self.buffer = []

    def sample(self, n = None, mode_sample = None):
        if self.mode_sample == "last":
            d = self.sample_last(n)
        elif self.mode_sample == "random":
            d = self.sample_random(n)
        elif mode_sample == "all":
            d = self.memory
        # d = self.tuple(*zip(*d))
        # d = [np.stack(x) for x in d]
        # d = [torch.from_numpy(x).float().to(self.device) for x in d]
        # d = self.tuple(*d)
        return d
    
    def sample_last(self, n = 1):
        return self.memory[-n:]
    
    def sample_random(self, n = 1):
        sub = random.sample(self.memory, n)
        return sub