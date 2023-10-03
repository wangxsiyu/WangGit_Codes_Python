from .W_Trainer_pipeline_base import W_trainer_pipeline_base
from .W_Worker import W_Worker
import yaml
import pandas as pd

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
        for i in range(len(self.recordplan)):
            rp = self.recordplan.loc[i]
            tenv = self.select_env_by_name(rp.env)
            self.load_model(self.config['model'][rp.modelname], tenv)
            self.worker = W_Worker(tenv, self.model, self.device)
            self.worker.reload_env(tenv)
            self.worker.load_latest_model(rp.modelfolder)
            self.worker.record(rp.savename, n_episode = rp.n_episode, is_record = rp.is_record *arg, **kwarg)

        

