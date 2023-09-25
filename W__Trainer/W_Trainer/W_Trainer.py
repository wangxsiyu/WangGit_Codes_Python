
from .W_Worker import W_Worker
from .W_Logger import W_Logger
from .W_loss import W_loss
from .W_Buffer import W_Buffer
from W_Python.W import W
from tqdm import tqdm 
import torch
import numpy as np
from collections import namedtuple 

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
        action_dist = torch.nn.functional.softmax(action_vector, dim = -1)
        # action_dist = action_dist.permute((1,0,2))
        eps = 1e-4
        action_dist = action_dist.clamp(eps, 1-eps)
        action_onehot = W.W_onehot_array(buffer.action.squeeze(), action_dist.shape[-1]).to(self.device)
        action_likelihood = (action_dist * action_onehot).sum(-1)
        tb = namedtuple('TrainingBuffer', ("action_dist", "action_likelihood","outputs"))
        return tb(action_dist, action_likelihood, buffer.additional_output)
    
    def resume_training(self, max_episodes, folder, is_resume = True, model_pretrained = None):
        if is_resume:
            [filename, start_episode] = self.find_latest_model(currentfolder = folder)
            if start_episode > 0:
                model_pretrained = filename
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
            self.buffer.load(supervised_data_path)
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
        