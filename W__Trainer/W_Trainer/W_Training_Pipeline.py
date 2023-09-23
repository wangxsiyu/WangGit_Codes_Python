from W_Env.W_Env import W_Env
from W_Trainer.W_Trainer import W_Trainer
from W_Python.W import W
import yaml
import torch
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import os

def getmodel(modelinfo, env, device):
    model_name = modelinfo['name']
    ldic = locals()
    exec(f"from W_RNN.{model_name} import {model_name} as W_model", globals(), ldic)
    W_model = ldic['W_model']
    info = modelinfo['param_model']
    info.update({'env': env})
    model = W_model(info_model = info, device = device)
    return model

class W_Training_Curriculum():
    def __init__(self, yaml_trainer = 'trainer.yaml', yaml_curriculum = 'curriculum.yaml'):        
        with open(yaml_trainer, 'r', encoding="utf-8") as fin:
            self.trainerinfo = yaml.load(fin, Loader=yaml.FullLoader)
        with open(yaml_curriculum, 'r', encoding="utf-8") as fin:
            curriculum = yaml.load(fin, Loader=yaml.FullLoader)
        self.n_course = len(curriculum)
        self.curriculum = []
        for i in range(self.n_course):
            self.curriculum += [curriculum[f'Course{i+1}']]
          
        config = self.trainerinfo
        device = config['device']
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.env = []
        for i in range(self.n_course):
            course = self.curriculum[i]
            env = W_Env(course['envname'], \
                    param_task = course['task'], \
                    param_metadata = course['metadata'], \
                    render_mode = None)
            self.env += [env]

        self.model = getmodel(config['model'], self.env[0], device = self.device)
        self.trainer = W_Trainer(None, self.model, param_loss = config['param_loss'], \
                            param_optim = config['param_optim'], \
                            param_logger = config['param_logger'], \
                            param_buffer = config['param_buffer'], \
                            gradientclipping = config['trainer']['max-grad-norm'], \
                            save_path = config['save_path'], \
                            device = self.device)
        yamlsavepath = W.W_mkdir(os.path.join(config['save_path'], 'config_files'))
        with open(os.path.join(yamlsavepath, "trainerinfo.yaml"), 'w') as fout:
            yaml.dump(self.trainerinfo, fout)
        with open(os.path.join(yamlsavepath, "curriculum.yaml"), 'w') as fout:
            yaml.dump(curriculum, fout)
        
    def train(self, seed = 0):
        is_resume = self.trainerinfo['trainer']['is_resume']
        self.trainer.setup_randomseed(seed)
        lastfile = None
        for coursei in range(self.n_course):
            savepath = os.path.join(self.trainerinfo['save_path'], f"Seed{seed}")
            savepath = W.W_mkdir(os.path.join(savepath, f"C{coursei}_{self.curriculum[coursei]['coursename']}"))
            self.trainer.reload_env(self.env[coursei])
            max_episodes = self.curriculum[coursei]['train_episodes']
            [currentfile, start_episode] = self.trainer.find_latest_model(savepath)
            if is_resume and start_episode == max_episodes: # done training
                lastfile = currentfile
                print(f"course {coursei} trained already: skip")
            else:
                print(f"training course {coursei}")
                self.trainer.train(savepath = savepath, max_episodes= max_episodes, \
                            batch_size= self.trainerinfo['trainer']['batch_size'], \
                            train_mode= self.trainerinfo['trainer']['train_mode'], \
                            is_online= self.trainerinfo['trainer']['is_online'], \
                            is_resume = is_resume, \
                            model_pretrained = lastfile)


