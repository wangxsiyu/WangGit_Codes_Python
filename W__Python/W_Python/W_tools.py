import numpy as np
def W_dict_deleteNone(**kwargs):
    res = dict((k,v) for k,v in kwargs.items() if v is not None)
    return res

def W_dict_kwargs():
    import inspect
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs

def enlist(a):
    if type(a) == type(np.array([])):
        a = a.tolist()
    if not type(a) == type([]):
        a = [a]
    return a

def W_dict_updateonly(dict1, dict2):
    dict1.update((k, dict2[k]) for k in dict1.keys() & dict2.keys())
    return dict1

def W_onehot(x, n):
    return np.squeeze(np.eye(n)[x])