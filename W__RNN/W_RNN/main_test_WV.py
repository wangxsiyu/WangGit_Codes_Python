import torch
import os
import torch
from W_RNN import W_RNN_Head_ActorCritic
from W_Trainer import W_Trainer, W_Worker
import yaml
import argparse
from W_Env.W_Env import W_Env
import numpy as np
import matplotlib.pyplot as plt

render_mode = None
n_maxT = 100
env = W_Env("WV", render_mode = render_mode, \
                        n_maxT = n_maxT)

with open('task.yaml', 'r', encoding="utf-8") as fin:
    config = yaml.load(fin, Loader=yaml.FullLoader)
device = torch.device("cpu")
model = W_RNN_Head_ActorCritic(env.observation_space_size() + env.action_space.n + 1,\
                           config['a2c']['mem-units'],env.action_space.n,'vanilla',device = device)
modeldata = torch.load('./md1/model_WV_5000.pt')
model.load_state_dict(modeldata['state_dict'])
wk = W_Worker(env, model, capacity = 1000,device = device)

wk.run_worker(1000)
buffer = wk.memory.sample(1000)
(obs,action,reward,timestep,done) = buffer

action = torch.matmul(action, torch.tensor([1,2,3], dtype = torch.float))
fixation = obs[:,:,0].squeeze()
image = torch.matmul(obs[:,:,1:10], torch.arange(1,10).float())
red = obs[:,:,10].squeeze()
purple = obs[:,:,11].squeeze()
green = obs[:,:,12].squeeze()

# divide into trials
d_images = []
d_choices = []
for i in range(action.shape[0]):
    tim = np.where(image[i,:] > 0)[0]
    tred = np.where(red[i,:] > 0)[0]
    # tpur = np.where(purple[i,:] > 0)[0]
    tim = tim[:len(tred)]
    timage = image[i, tim]
    c = action[i, tred]
    d_images.append(timage)
    d_choices.append(c)
ii = torch.cat(d_images)
cc = torch.cat(d_choices)

p = [np.mean((cc[ii == x + 1] == 3).numpy()) for x in range(9)]

od = np.array([3,2,6,9,5,8,1,7,4])
print([p[x-1] for x in od])