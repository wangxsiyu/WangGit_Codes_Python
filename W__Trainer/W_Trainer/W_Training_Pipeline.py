from .W_Trainer import W_Trainer
from .W_Trainer_pipeline_base import W_trainer_pipeline_base
from W_Python.W import W
import yaml
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import os
import copy


class W_Training_Curriculum(W_trainer_pipeline_base):
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
        self.load_device(config['device'])
        
        self.env = []
        for i in range(self.n_course):
            course = self.curriculum[i]
            env = self.import_env(course)
            self.env += [copy.deepcopy(env)]

        self.load_model(config['model'], self.env[0])

        save_path = config['save_path']
        self.trainer = W_Trainer(None, self.model, param_loss = config['param_loss'], \
                            param_optim = config['param_optim'], \
                            param_logger = config['param_logger'], \
                            param_buffer = config['param_buffer'], \
                            gradientclipping = config['trainer']['max-grad-norm'], \
                            save_path = save_path, \
                            device = self.device)
        yamlsavepath = W.W_mkdir(os.path.join(config['save_path'], 'config_files'))
        with open(os.path.join(yamlsavepath, "trainerinfo.yaml"), 'w') as fout:
            yaml.dump(self.trainerinfo, fout)
        with open(os.path.join(yamlsavepath, "curriculum.yaml"), 'w') as fout:
            yaml.dump(curriculum, fout)
        
    def train(self, seed = 0):
        is_resume = self.trainerinfo['trainer']['is_resume']
        if 'is_online' in self.trainerinfo['trainer'].keys():
            is_online = self.trainerinfo['trainer']['is_online']
        else:
            is_online = False
        lastfile = None
        modelname = self.trainerinfo['model']['name'] + '_' + self.trainerinfo['model']['param_model']['gatetype']
        for coursei in range(self.n_course):
            self.trainer.setup_randomseed(seed)
            savepath = os.path.join(self.trainerinfo['save_path'], f"Seed{seed}_{modelname}")
            savename = f"C{coursei+1}_{self.curriculum[coursei]['coursename']}_{self.trainerinfo['trainer']['train_mode']}"
            if is_online:
                savename = savename + "_online"
            savepath = W.W_mkdir(os.path.join(savepath, savename))
            self.trainer.reload_env(self.env[coursei])
            max_episodes = self.curriculum[coursei]['train_episodes']
            [currentfile, start_episode] = self.trainer.find_latest_model(savepath)
            if is_resume and start_episode == max_episodes: # done training
                lastfile = currentfile
                print(f"course {coursei} trained already: skip")
            else:
                print(f"training course {coursei}")
                if 'supervised_data_path' in self.trainerinfo['trainer'].keys():
                    supervised_data_path = self.trainerinfo['trainer']['supervised_data_path']
                else:
                    supervised_data_path = None
                self.trainer.train(savepath = savepath, max_episodes= max_episodes, \
                            batch_size= self.trainerinfo['trainer']['batch_size'], \
                            train_mode= self.trainerinfo['trainer']['train_mode'], \
                            is_online= is_online, \
                            is_resume = is_resume, \
                            model_pretrained = lastfile, \
                            supervised_data_path = supervised_data_path)


