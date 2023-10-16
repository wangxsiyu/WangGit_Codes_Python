from .W_Trainer_pipeline_base import W_trainer_pipeline_base
from .W_Worker import W_Worker
import yaml
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
import copy
import numpy as np
import matplotlib.pyplot as plt

class W_record(W_trainer_pipeline_base):
    def __init__(self, yaml_setup = 'record_setup.yaml', file_record = 'record_plan.csv'):     
        with open(yaml_setup, 'r', encoding="utf-8") as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)
        self.config = config  
        self.load_envs(config['envs'])
        self.load_device(config['device'])
        self.worker = None  
        self.recordplan = pd.read_csv(file_record, sep = ',')

    def record(self, *arg, **kwarg):
        num_cores = multiprocessing.cpu_count()
        print(f'ncore = {num_cores}')
        njob = len(self.recordplan)
        for i in range(njob):
            self.record_singlethread(i, *arg, **kwarg)
        # Parallel(n_jobs=njob)(delayed(function=self.record_singlethread)(i, *arg, **kwarg) for i in range(njob))

    def record_singlethread(self, i, *arg, **kwarg):
        rp = self.recordplan.loc[i]
        tenv = copy.deepcopy(self.select_env_by_name(rp.env))
        model = self.load_model(self.config['model'][rp.modelname], tenv, is_auto_set = False)
        worker = W_Worker(tenv, model, self.device)
        worker = copy.deepcopy(worker)
        # self.worker.reload_env(tenv)

        if 'modeliter' in self.recordplan.columns:
            worker.load_folder_model(rp.modelfolder, rp.modeliter)
        else:
            worker.load_latest_model(rp.modelfolder)
        worker.record(rp.savename, n_episode = rp.n_episode, is_record = rp.is_record, *arg, **kwarg)

    def plot_loss(self):
        njob = len(self.recordplan)
        for i in range(njob):
            rp = self.recordplan.loc[i]
            tenv = copy.deepcopy(self.select_env_by_name(rp.env))
            model = self.load_model(self.config['model'][rp.modelname], tenv, is_auto_set = False)
            worker = W_Worker(tenv, model, self.device)
            info = worker.load_model(worker.find_latest_model(rp.modelfolder)[0])
            ls = np.array([x['reward'] for x in info['info']])
            ls = np.convolve(ls, np.ones((1000,))/1000, mode='same') 
            plt.plot(np.arange(len(ls)), ls)
        plt.ylim(0, 300)
        plt.xlabel('training episode')
        plt.ylabel('average reward per block')
        plt.show()

        

        

