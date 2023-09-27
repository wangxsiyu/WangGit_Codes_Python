from .W_Trainer_pipeline_base import W_trainer_pipeline_base
from .W_Worker import W_Worker
import yaml

class W_record(W_trainer_pipeline_base):
    def __init__(self, yaml_record = 'record.yaml'):     
        with open(yaml_record, 'r', encoding="utf-8") as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)
        self.config = config
        self.load_device(config['device'])
        self.env = self.import_env(config['env'])
        self.load_model(config['model'], self.env)
        self.worker = W_Worker(self.env, self.model, self.device)

    def record(self, *arg, **kwarg):
        self.worker.record(self.config['savename'], self.config['n_episode'], *arg, **kwarg)

        

