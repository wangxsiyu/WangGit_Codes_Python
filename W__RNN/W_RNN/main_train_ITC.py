import torch
from W_RNN import W_RNN_Head_ActorCritic
from W_Trainer import W_Trainer
import yaml
import argparse
from W_Env.W_Env import W_Env

render_mode = None
n_maxT = 100
env = W_Env("TwoStep_1frame", render_mode = render_mode, \
                        n_maxT = n_maxT)
parser = argparse.ArgumentParser(description="parameters")
parser.add_argument('-c','--config', type = str, default = 'task.yaml')
args = parser.parse_args()

with open(args.config, 'r', encoding="utf-8") as fin:
    config = yaml.load(fin, Loader=yaml.FullLoader)

model = W_RNN_Head_ActorCritic(env.observation_space_size() + env.action_space.n + 1,\
                           config['a2c']['mem-units'],env.action_space.n,'vanilla')
loss = dict(name = 'A2C', params = dict(gamma = config['a2c']['gamma'], \
                                        coef_valueloss = config['a2c']['value-loss-weight'], \
                                        coef_entropyloss = config['a2c']['entropy-loss-weight']))
optim = dict(name = 'RMSprop', lr  = config['a2c']['lr'])
wk = W_Trainer(env, model, loss, optim, capacity = 100, mode_sample = "last")
wk.train(10000, batch_size=1)
# wk.train(30000
# , save_path='./md/model_WV', save_interval= 1000)