import torch
from collections import namedtuple 
import numpy as np
import random
from W_Python import W_tools as W
from tqdm import tqdm 
import pickle
import os
import re
import pandas

class W_Buffer:
    def __init__(self, tuple, capacity = np.Inf, mode_sample = "random", device = None, *arg, **kwarg):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"enabling {self.device}")
        else:
            self.device = device
        self.memory = []
        self.tuple = tuple
        self.capacity = capacity
        self.mode_sample = mode_sample
        self.clear()

    def clear(self):
        self.buffer = []

    def add(self, *arg):
        d = [np.array(x, dtype = 'float') for x in arg]
        d = self.tuple(*d)
        self.buffer += [d]
        
    def push(self):
        buffer = self.tuple(*zip(*self.buffer))
        buffer = [np.concatenate(x) for x in buffer]
        buffer = self.tuple(*buffer)
        self.memory += [buffer]
        while len(self.memory) > self.capacity:
            self.memory.pop(0)
        self.buffer = []

    def get_last(self, n = 1):
        return self.memory[-n:]
    
    def get_random(self, n = 1):
        sub = random.sample(self.memory, n)
        return sub

    def sample(self, n = None, mode_sample = None):
        if mode_sample is None:
            mode_sample = self.mode_sample
        if mode_sample == "last":
            d = self.get_last(n)
        elif mode_sample == "random":
            d = self.get_random(n)
        elif mode_sample == "all":
            d = self.memory
        d = self.tuple(*zip(*d))
        d = [np.stack(x) for x in d]
        d = [torch.from_numpy(x).float().to(self.device) for x in d]
        d = self.tuple(*d)
        return d

class W_Worker:
    def __init__(self, env, model, device = None, *arg, **kwarg):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"enabling {self.device}")
        else:
            self.device = device
        self.env = env
        self.model = model
        Memory = namedtuple('Memory', ('obs', 'action', 'reward', 'timestep', 'done'))
        self.memory = W_Buffer(Memory, device = device, *arg, **kwarg)
        # self.set_mode(*arg, **kwarg)

    def set_mode(self, mode_worker = "Test"):
        self.mode_worker = mode_worker

    def select_action(self, action_vector, mode_action):
        if mode_action == "softmax":
            action_dist = torch.nn.functional.softmax(action_vector, dim = -1)
            action_cat = torch.distributions.Categorical(action_dist.squeeze())
            action = action_cat.sample()
        return action

    def run_episode(self, mode_action = "softmax", is_test = False, record = False):
        if self.model is not None:
            self.model.eval()
        done = False
        total_reward = 0
        obs = self.env.reset()
        mem_state = None
        if not is_test:
            self.memory.clear()
        recording = []
        while not done:
            # take actions
            obs = torch.from_numpy(obs).unsqueeze(0).float()
            if self.model is not None:
                action_vector, val_estimate, mem_state = self.model(obs.unsqueeze(0).to(self.device), mem_state)
                action = self.select_action(action_vector, mode_action)
            elif mode_action == "oracle":            
                action = torch.tensor(self.env.get_oracle_action(), dtype = torch.int64)
            if record:
                recording.append(mem_state[0])
            obs_new, reward, done, timestep, _ = self.env.step(action.item())
            reward = float(reward)
            
            if not is_test:
                action_onehot = torch.nn.functional.one_hot(action, self.env.action_space.n)
                action_onehot = action_onehot.unsqueeze(0).float()
                self.memory.add(obs.to('cpu').numpy(), action_onehot.to('cpu').numpy(), [reward], [timestep], [done])
            
            obs = obs_new
            total_reward += reward
        if record:
            recording = torch.concatenate(recording).squeeze()
       
        if not is_test:
            self.memory.push()
        if is_test:
            return self.env.format4save(), recording
        else:
            return total_reward, self.env._get_info()

    def run_episode_outputlayer(self, buffer, safeoption = 'all'):
        self.model.train()
        (obs, action, _,_,_) = buffer
        action = action.to(self.device)
        mem_state = None
        action_vector, val_estimate, mem_state = self.model(obs.to(self.device), mem_state)
        action_dist = torch.nn.functional.softmax(action_vector, dim = -1)
        action_dist = action_dist.permute((1,0,2))
        eps = 1e-4
        action_dist = action_dist.clamp(eps, 1-eps)
        val_estimate = val_estimate.permute((1,0,2)).squeeze(2)
        action_likelihood = (action_dist * action).sum(-1)
        tb = namedtuple('TrainingBuffer', ("action_dist","value", "action_likelihood"))
        return tb(action_dist, val_estimate, action_likelihood)
        
    def run_worker(self, nrep = 1, showprogress = False, savename = None, *arg, **kwarg):
        # W.W_tic()
        rs = []
        rg = range(nrep)
        if showprogress:
            rg = tqdm(rg)
        for i in rg:
            r, lastinfo = self.run_episode(*arg, **kwarg)
            # r.blockID = np.ones_like(r.blockID) * (i+1) 
            rs.append(r)
        # W.W_toc("worker time = ")
        if savename is None:
            return np.mean(rs), lastinfo
        else:
            d = pandas.concat(rs)
            d.to_csv(savename)
            if lastinfo is not None:
                efs = lastinfo.numpy()
                efs = pandas.DataFrame(efs)
                efs.to_csv(savename.replace('data', 'efs'))

    def run_oracle(self, nepisode, filename = None, *arg, **kwarg):
        self.run_worker(nepisode, showprogress= True, mode_action = "oracle", *arg, **kwarg)
        memory = self.memory.sample(mode_sample="all")
        if filename is not None:
            with open(filename, "wb") as f:
                pickle.dump([memory.obs, memory.action, memory.reward, memory.timestep, memory.done], f)

        
    def loaddict_folder(self, currentfolder, is_resume = True):
        isload = False
        file_trained_list = os.listdir(currentfolder)
        fs = [re.search("(.*)_(.*).pt", x) for x in file_trained_list]
        fs = [x for x in fs if x is not None]
        if not len(fs) == 0:
            its = [x.group(2) for x in fs]
            tid = np.argmax([int(x) for x in its])
            isload = self.loaddict(os.path.join(currentfolder, fs[tid].group(0)), is_resume)
        return isload
    
    def loaddict(self, loadname, is_resume = True):
        isload = False
        if loadname is not None:
            modeldata = torch.load(loadname)
            self.model.load_state_dict(modeldata['state_dict'])
            print(f"loaded model {loadname}")
            if hasattr(self, 'logger'):
                self.logger.last_saved_version = loadname
                if is_resume:
                    self.logger.start_episode = int(re.search("(.*)_(.*).pt", loadname).group(2)) + 1
                    self.logger.setlog(modeldata['training_info'])
            isload = True
        return isload
    
    def loaddict_main(self, loadname = None, currentfolder = None, isresume = False):
        isload = False
        if currentfolder is not None:
            isload = self.loaddict_folder(currentfolder, True)
        if not isload and loadname is not None:
            isload = self.loaddict(loadname, isresume)