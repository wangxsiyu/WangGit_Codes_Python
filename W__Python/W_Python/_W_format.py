import numpy as np

class W_format():
    def enlist(a):
        if type(a) == type(np.array([])):
            a = a.tolist()
        if not type(a) == type([]):
            a = [a]
        return a