
from .W_Worker import W_Worker
from .W_Logger import W_Logger
from .W_loss import W_loss
from .W_Buffer import W_Buffer
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

    def setup_randomseed(self, seed):
        if seed is not None:    
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
        action_likelihood = (action_dist * buffer.action).sum(-1)
        tb = namedtuple('TrainingBuffer', ("action_dist","value", "action_likelihood"))
        return tb(action_dist, val_estimate, action_likelihood)
    
    def train(self, max_episodes = 10, batch_size = 1, train_mode = "RL", is_online = False, \
              progressbar_position = 0, *arg, **kwarg):
        tqdmrange = self.logger.initialize(max_episodes)
        progress = tqdm(tqdmrange, position = progressbar_position, leave=True)
        reward, newdata = self.train_generatedata(batch_size, train_mode, is_online)
        self.logger.update(reward, None, newdata)
        for _ in progress:
            batchdata = self.train_getdata(batch_size, train_mode, is_online)
            modelbuffer = self.run_episode_buffer(batchdata)
            self.optimizer.zero_grad()
            loss, info_loss = self.loss.loss(batchdata, modelbuffer)
            loss.backward()
            if self.gradientclipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradientclipping)
            self.optimizer.step()
            # if tqdmstr is None:
            #     tqdmstr = f"Process {tqdmpos}"
            # progress.set_description(f"{tqdmstr}, {self.logger.getdescription()}, Loss: {loss.item():.4f}")
            reward, newdata = self.train_generatedata(batch_size, train_mode, is_online)
            self.logger.update(reward, info_loss, newdata)
            self.logger.save(self.model.state_dict())
                
    def train_generatedata(self, batch_size, train_mode, is_online, *arg, **kwarg):
        if train_mode == "RL":
            if is_online: # create new experience for each batch
                self.buffer.clear_buffer()
                reward, data = self.work(n_episode = batch_size, *arg, **kwarg)
            else: # create 1 experience, and resample from the memory buffer
                reward, data = self.work(n_episode = 1, *arg, **kwarg)
        elif train_mode == "supervised":
            if self.logger.supervised_test_interval is not None and self.logger.episode % self.logger.supervised_test_interval == 0:
                reward, data = self.work(n_episode = 1, *arg, **kwarg)
            else:
                reward = np.NaN
                data = None
        self.buffer.push(data)
        return reward, data

    def train_getdata(self, batch_size, train_mode, is_online, *arg, **kwarg):
        if train_mode == "RL":          
            if is_online:
                batchdata = self.buffer.sample(batch_size, 'all')
            else:
                batchdata = self.buffer.sample(batch_size)
        elif train_mode == "supervised":
            batchdata = self.buffer.sample(batch_size)
            # tid = np.random.choice(self.training_memory.reward.shape[0], batch_size)
            # buffer = [x[tid] for x in self.training_memory]
            # buffer = self.memory.tuple(*buffer)
        return batchdata
    
    def load_buffer_from_data(self, filename):
        import pickle
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.buffer.push(data)
        