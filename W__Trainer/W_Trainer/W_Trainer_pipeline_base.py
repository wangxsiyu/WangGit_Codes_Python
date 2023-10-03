from W_Env.W_Env import W_Env
from W_Python.W import W
import torch

def getmodel(modelinfo, env = None, device = None):
    model_name = modelinfo['name']
    ldic = locals()
    exec(f"from W_RNN.{model_name} import {model_name} as W_model", globals(), ldic)
    W_model = ldic['W_model']
    info = modelinfo['param_model'].copy()
    info.update({'env': env})
    model = W_model(info_model = info, device = device)
    return model

class W_trainer_pipeline_base():
    def __init__(self) -> None:
        pass

    def load_device(self, device = "auto"):
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def load_model(self, config_model, env = None):
        self.model = getmodel(config_model, env, device = self.device)

    def import_env(self, config_env):
        env = W_Env(config_env['envname'], \
                    param_task = config_env['task'], \
                    param_metadata = config_env['metadata'], \
                    render_mode = None)
        return env
    
    def select_env_by_name(self, name):
        return self.envs[W.W_list_findidx(name, self.envnames)]

    def load_envs(self, config):
        self.envs = []
        self.envnames = list(config.keys())
        for i in range(len(self.envnames)):
            env = self.import_env(config[self.envnames[i]])
            self.envs += [env]
