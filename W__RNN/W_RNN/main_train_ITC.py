import torch
from W_RNN import W_RNN_Head_ActorCritic
from W_Trainer import W_Trainer
import yaml
import argparse
from W_Env.W_Env import W_Env
import numpy as np

render_mode = None
n_maxT = 200
env = W_Env("WV", render_mode = render_mode, \
                        n_maxT = n_maxT)
parser = argparse.ArgumentParser(description="parameters")
parser.add_argument('-c','--config', type = str, default = 'task.yaml')
args = parser.parse_args()

with open(args.config, 'r', encoding="utf-8") as fin:
    config = yaml.load(fin, Loader=yaml.FullLoader)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"
device = torch.device(device)
print(f"enabling {device}")



n_seeds = 8
for seed_idx in range(1, n_seeds + 1):
    tseed = 1995 * seed_idx
    tlt = "v" + f"_{seed_idx}"
    import os
    exp_path = os.path.join("md2", tlt)
    if not os.path.isdir(exp_path): 
        os.mkdir(exp_path)
    print("running" + f"_{seed_idx}")
    noise_scale = 0.05 if np.random.rand() < 0.5 else 0
    print(f"noise_scale = {noise_scale}")
    config['noise_scale'] = noise_scale    
    out_path = os.path.join(exp_path, f"train_iter")
    with open(out_path + ".yaml", 'w') as fout:
        yaml.dump(config, fout)

    
    model = W_RNN_Head_ActorCritic(env.observation_space_size() + env.action_space.n + 1,\
                            config['a2c']['mem-units'],env.action_space.n,'noise',noise_scale = noise_scale, device = device)
    loss = dict(name = 'A2C', params = dict(gamma = config['a2c']['gamma'], \
                                            coef_valueloss = config['a2c']['value-loss-weight'], \
                                            coef_entropyloss = config['a2c']['entropy-loss-weight']))
    optim = dict(name = 'RMSprop', lr  = config['a2c']['lr'])
    wk = W_Trainer(env, model, loss, optim, capacity = 1000, mode_sample = "last", device = device, gradientclipping=config['a2c']['max-grad-norm'])
    wk.train(2000, batch_size = 32, save_path= out_path, save_interval= 200)