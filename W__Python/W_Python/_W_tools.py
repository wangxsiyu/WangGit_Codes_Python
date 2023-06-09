
import numpy as np
import itertools

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
        return np.squeeze(np.eye(n)[x])
