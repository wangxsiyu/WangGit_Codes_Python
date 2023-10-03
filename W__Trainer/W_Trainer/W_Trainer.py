
from .W_Worker import W_Worker
from .W_Logger import W_Logger
from .W_loss import W_loss
from .W_Buffer import W_Buffer
from W_Python.W import W
from tqdm import tqdm 
import torch
import pandas as pd
import numpy as np
from collections import namedtuple 
import os
import pickle        

class W_Trainer(W_Worker): 
    # env
    # model
    # loss
    # logger
    # optimizer
    # device (for training)
    seed = None
    gradientclipping = None
    def __init__(self, env, model, param_loss, param_optim, param_logger = None, param_buffer = None, \
                 gradientclipping = None, \
                 seed = None, *arg, **kwarg):
        super().__init__(env, model, *arg, **kwarg)
        self.setup_randomseed(seed)
        self.gradientclipping = gradientclipping
        self.loss = W_loss(param_loss, device = self.device)
        self.buffer = W_Buffer(param_buffer, device = self.device)
        self.logger = W_Logger(param_logger)
        self.setup_optimizer(param_optim)

    def setup_randomseed(self, seed = None):
        if seed is None:
            seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        self.seed = seed
        
    def setup_optimizer(self, param_optim):
        params = list(self.model.parameters())
        # params += list(self.model.init0)
        if param_optim['name'] == "RMSprop":
            self.optimizer = torch.optim.RMSprop(params, lr = param_optim['lr'])

    def run_episode_buffer(self, buffer, *arg, **kwarg): # NEED EDIT
        self.model.train()
        # (obs, action, _,_,_) = buffer
        if hasattr(self.model, 'initialize_latentvariables'):
            LV = self.model.initialize_latentvariables()
        else:
            LV = None
        action_vector, LV, additional_output = self.model(buffer.obs, LV)
        if len(action_vector.shape) == 2: # unbatched
            action_vector = action_vector.unsqueeze(0)
            additional_output = additional_output.unsqueeze(0)
        action_dist = torch.nn.functional.softmax(action_vector, dim = -1)
        # action_dist = action_dist.permute((1,0,2))
        eps = 1e-4
        action_dist = action_dist.clamp(eps, 1-eps)
        action_onehot = W.W_onehot_array(buffer.action.squeeze(-1), action_dist.shape[-1]).to(self.device)
        action_likelihood = (action_dist * action_onehot).sum(-1)
        tb = namedtuple('TrainingBuffer', ("action_dist", "action_likelihood","outputs"))
        return tb(action_dist, action_likelihood, additional_output)
    
    def resume_training(self, max_episodes, folder, is_resume = True, model_pretrained = None):
        if is_resume:
            [info, _, start_episode] = self.load_latest_model(folder)
        else:        
            start_episode = 0
            info = self.load_model(model_pretrained)
        return self.logger.initialize(max_episodes, start_episode, info)

    def train(self, savepath = '', max_episodes = 10, batch_size = 1, \
              train_mode = "RL", supervised_data_path = None, \
              is_online = False, is_resume = True, model_pretrained = None, \
              progressbar_position = 0, *arg, **kwarg):
        self.buffer.clear_buffer()
        if train_mode == "supervised":
            self.train_load_supervised(supervised_data_path)
        tqdmrange = self.resume_training(max_episodes, savepath, is_resume = is_resume, model_pretrained = model_pretrained)
        if len(tqdmrange) == 0:
            print(f'model already trained: total steps = {max_episodes}, skip')
            return
        if progressbar_position is not None:
            progress = tqdm(tqdmrange, position = progressbar_position, leave=True)
        else:
            progress = tqdmrange
        reward, newdata = self.train_generatedata(batch_size, train_mode, is_online)
        # progress.set_description(f"{train_mode}, {self.logger.getdescription()}, start", refresh = False)
        self.logger.update0(reward, None, newdata)
        self.logger.save(savepath, self.model.state_dict())
        for _ in progress:
            batchdata = self.train_getdata(batch_size, train_mode, is_online)
            modelbuffer = self.run_episode_buffer(batchdata)
            self.optimizer.zero_grad()
            loss, info_loss = self.loss.loss(batchdata, modelbuffer)
            loss.backward()
            if self.gradientclipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradientclipping)
            self.optimizer.step()
            progress.set_description(f"{train_mode}, {self.logger.getdescription()}, Loss: {loss.item():.4f}", refresh = False)
            reward, newdata = self.train_generatedata(batch_size, train_mode, is_online)
            self.logger.update1(reward, info_loss, newdata)
            self.logger.save(savepath, self.model.state_dict())
        # progress.set_description(f"{train_mode}, {self.logger.getdescription()}, Loss: {loss.item():.4f}, complete", refresh = True)
        progress.close()

    def train_generatedata(self, batch_size, train_mode, is_online, *arg, **kwarg):
        if train_mode == "RL":
            if is_online: # create new experience for each batch
                self.buffer.clear_buffer()
                reward, data = self.work(n_episode = batch_size, *arg, **kwarg)
            else: # create 1 experience, and resample from the memory buffer
                if len(self.buffer.memory) == 0:
                    reward, data = self.work(n_episode = batch_size, *arg, **kwarg)
                else:
                    reward, data = self.work(n_episode = 1, *arg, **kwarg) 
            self.buffer.push(data)                   
        elif train_mode == "supervised":
            if self.logger.metadata_logger['supervised_test_interval'] is not None and \
                self.logger.episode % self.logger.metadata_logger['supervised_test_interval'] == 0:
                reward, data = self.work(n_episode = 1, *arg, **kwarg)
            else:
                reward = np.NaN
                data = None
        return reward, data

    def train_load_supervised(self, file): 
        file_preload = os.path.splitext(file)[0] + ".pkl"
        if os.path.exists(file_preload):
            with open(file_preload, 'rb') as f:
                data = pickle.load(f)
        else:
            data = self.loadandsave_supervised(file) 
        self.buffer.push(data)   
        
    def loadandsave_supervised(self, file):
        d = pd.read_csv(file)
        colnames = list(d.keys())
        episodes = np.unique(d.episodeID)
        n_episodes = len(episodes)
        data = []
        progress_load = tqdm(range(n_episodes))
        progress_load.set_description(f'loading supervised data')
        for i in progress_load:
            rid = d.episodeID == episodes[i]
            td = d.loc[rid,:]
            tdata = self.train_load_supervised_episode(td.to_dict('records'))
            data.append(tdata)
        progress_load.close()
        
        savename = os.path.splitext(file)[0] + ".pkl"
        with open(savename, 'wb') as f:
            pickle.dump(data, f)
        return data

    def train_load_supervised_episode(self, superviseddata):
        done = False
        data = {'obs':[], 'reward':[], 'action':[], 'obs_next':[], 'timestep':[], \
                'isdone': [], 'additional_output':[]}
        superviseddata[0]['last_action'] = None
        superviseddata[0]['last_reward'] = 0
        obs, additional_output = self.env.format_supervised(superviseddata[0])
        obs = torch.from_numpy(obs).unsqueeze(0).float()
        for i in range(len(superviseddata)):
            data['obs'].append(obs)
            data['additional_output'].append(additional_output)

            if i + 1 < len(superviseddata):
                superviseddata[i+1]['last_action'] = superviseddata[i]['action']
                superviseddata[i+1]['last_reward'] = superviseddata[i]['reward'] 
                obs_next, additional_output = self.env.format_supervised(superviseddata[i+1])
                obs_next = torch.from_numpy(obs_next).unsqueeze(0).float()
            else:
                obs_next = obs
                additional_output = None
            data['obs_next'].append(obs_next)
            obs = obs_next
            
            action = superviseddata[i]['action']
            data['action'].append(torch.tensor(action).unsqueeze(0).unsqueeze(0))

            reward = superviseddata[i]['reward']
            data['reward'].append(torch.tensor(float(reward)).unsqueeze(0).unsqueeze(0))
            
            if 'timestep' in superviseddata[i]:
                timestep = superviseddata[i]['timestep']
            else:
                timestep = i
            data['timestep'].append(torch.tensor(timestep).unsqueeze(0).unsqueeze(0))

            if 'isdone' in superviseddata[i]:
                isdone = superviseddata[i]['isdone']
            else:
                isdone = 1 if i + 1 == len(superviseddata) else 0
            data['isdone'].append(torch.tensor(isdone).unsqueeze(0).unsqueeze(0))

        keys = list(data.keys())
        for x in keys:
            if all([None == i for i in data[x]]):
                # print(f"no {x}: skipped")
                data.pop(x)
            else:
                data[x] = torch.concat(data[x]).float()

        return data

    def train_getdata(self, batch_size, train_mode, is_online, *arg, **kwarg):
        if train_mode == "RL":          
            if is_online:
                batchdata = self.buffer.sample(batch_size, 'all')
            else:
                batchdata = self.buffer.sample(batch_size)
        elif train_mode == "supervised":
            batchdata = self.buffer.sample(batch_size, 'random')
            # tid = np.random.choice(self.training_memory.reward.shape[0], batch_size)
            # buffer = [x[tid] for x in self.training_memory]
            # buffer = self.memory.tuple(*buffer)
        return batchdata
    
    def load_buffer_from_data(self, filename):
        import pickle
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.buffer.push(data)
        