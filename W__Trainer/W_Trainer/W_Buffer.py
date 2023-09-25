import numpy as np
import random 
from W_Python.W import W

class W_Buffer:
    memory = []
    def __init__(self, param_buffer, device = None, *arg, **kwarg):
        self.device = device
        param = {"capacity": np.Inf, "mode_sample": "random"}
        param.update(param_buffer)
        self.capacity = param_buffer['capacity']
        self.mode_sample = param_buffer['mode_sample']
        self.clear_buffer()

    def clear_buffer(self):
        self.memory = []
    
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
        out = W.W_list_of_dict_to_namedtuple(d)
        return out
    
    def sample_last(self, n = 1):
        memory = self.memory[-n:]
        assert len(memory) == n
        return memory
    
    def sample_random(self, n = 1):
        memory = random.sample(self.memory, n)
        return memory