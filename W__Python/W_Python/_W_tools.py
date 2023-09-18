
import numpy as np
import itertools
import torch

class W_tools():
    def W_counter_balance(params):
        names = [k for k in params]
        pars = [params[k] for k in params]
        lst = list(itertools.product(*pars))  
        lst = np.array(lst)

        dct = dict(((names[k], list(lst[:,k])) for k in range(lst.shape[1])))
        dct['n'] = lst.shape[0]
        return dct

    def W_onehot(x, n):
        if torch.is_tensor(x):
            if x is None:
                return torch.zeros(n)
            else:
                return torch.squeeze(torch.eye(n)[x.cpu().numpy()])
        else:
            if x is None:
                return np.zeros(n)
            else:
                return np.squeeze(np.eye(n)[x])
