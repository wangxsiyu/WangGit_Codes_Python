from W_RNN.W_Worker import W_Worker
from W_RNN.W_Logger import W_Logger 
import torch
import numpy as np
from tqdm import tqdm 
# from parallelbar import progress_map as tqdm
from W_Python import W_tools as W
import os
import re
import pickle


class W_loss:
    loss_name = None
    def __init__(self, loss, device):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"enabling {self.device}")
        else:
            self.device = device
        
        self.loss_cross_entropy = torch.nn.CrossEntropyLoss()
        
        self.loss_name = loss['name']
        self.params = loss['params']

    def loss(self, *arg, **kwarg):
        if self.loss_name == "A2C":
            return self.loss_A2C(*arg, **kwarg)   
        elif self.loss_name == "A2C_supervised":
            return self.loss_A2C_supervised(*arg, **kwarg)     

    def loss_A2C_supervised(self, buffer, trainingbuffer):
        gamma = self.params['gamma']
        (_, action, reward, _, done) = buffer
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        (action_dist, value, action_likelihood) = trainingbuffer
        # bootstrap discounted returns with final value estimates
        nbatch = done.shape[0]
        nstep = done.shape[1]
        returns = torch.zeros((nbatch,)).to(self.device)
        all_returns = torch.zeros((nbatch, nstep)).to(self.device)
        # run Generalized Advantage Estimation, calculate returns, advantages
        for t in reversed(range(nstep)):
            mask = 1 - done[:,t]
            returns = reward[:,t] + returns * gamma * mask
            all_returns[:,t] = returns 

        policy_loss = self.loss_cross_entropy(action_dist.permute(0,2,1),action.permute(0,2,1))
        value_loss = 0.5 * (all_returns - value).pow(2).mean()
        entropy_reg = -(action_dist * torch.log(action_dist)).mean()

        loss_actor = policy_loss - self.params['coef_entropyloss'] * entropy_reg
        loss_critic = self.params['coef_valueloss'] * value_loss  
        loss = loss_actor + loss_critic

        if torch.isinf(loss):
            print('check')
        perr = 1 - torch.mean((action_dist.detach().argmax(axis = 2) == action.argmax(axis = 2)).float())
        info = {'perror': perr}
        return loss, info

    def loss_A2C(self, buffer, trainingbuffer):
        gamma = self.params['gamma']
        (_, _, reward, _, done) = buffer
        reward = reward.to(self.device)
        done = done.to(self.device)
        (action_dist, value, action_likelihood) = trainingbuffer
        # bootstrap discounted returns with final value estimates
        nbatch = done.shape[0]
        nstep = done.shape[1]
        returns = torch.zeros((nbatch,)).to(self.device)
        advantages = torch.zeros((nbatch,)).to(self.device)
        last_value = torch.zeros((nbatch,)).to(self.device)
        all_returns = torch.zeros((nbatch, nstep)).to(self.device)
        all_advantages = torch.zeros((nbatch, nstep)).to(self.device)
        # run Generalized Advantage Estimation, calculate returns, advantages
        for t in reversed(range(nstep)):
            mask = 1 - done[:,t]
            returns = reward[:,t] + returns * gamma * mask
            deltas = reward[:,t] + last_value * gamma * mask - value[:,t].data
            
            advantages = advantages * gamma * mask + deltas

            all_returns[:,t] = returns 
            all_advantages[:,t] = advantages
            last_value = value[:,t].data

        logll = torch.log(action_likelihood)
        policy_loss = -(logll * all_advantages).mean()
        value_loss = 0.5 * (all_returns - value).pow(2).mean()
        entropy_reg = -(action_dist * torch.log(action_dist)).mean()

        loss_actor = policy_loss - self.params['coef_entropyloss'] * entropy_reg
        loss_critic = self.params['coef_valueloss'] * value_loss  
        loss = loss_actor + loss_critic

        if torch.isinf(loss):
            print('check')
        return loss, None

class W_Trainer(W_Worker): 
    # env
    # model
    # loss
    # logger
    # optimizer
    # device (for training)
    def __init__(self, env, model, param_loss, param_optim, logger = None, device = None, gradientclipping = None, \
                 seed = None, *arg, **kwarg):
        super().__init__(env, model, device=device, *arg, **kwarg)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"enabling {self.device}")
        else:
            self.device = device
        if seed is not None:    
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.random.manual_seed(seed)
            self.seed = seed
        else:
            self.seed = 0
        self.set_mode(mode_worker = "train")
        self.loss = W_loss(param_loss, device = device)
        self.set_optim(param_optim)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = W_Logger()
        self.gradientclipping = gradientclipping
        self.logger.is_supervised = False
        self.logger.supervised_test_interval = 1

    def set_optim(self, param_optim):
        params = list(self.model.parameters())
        # params += list(self.model.init0)
        if param_optim['name'] == "RMSprop":
            self.optimizer = torch.optim.RMSprop(params, lr = param_optim['lr'])
    

    def train(self, max_episodes, batch_size, is_online = True, tqdmpos = 0, tqdmstr = None):
        self.logger.init(max_episodes)
        if self.logger.get_start_episode() >= max_episodes:
            print(f'model already trained: total steps = {max_episodes}, skip')
            return
        progress = tqdm(range(self.logger.get_start_episode()+1, max_episodes+1), position = tqdmpos, leave=True)
        self.progressbar = progress
        if not self.logger.is_supervised:
            reward, lastinfo = self.run_worker(batch_size)
            gamelen = len(self.memory.memory[-1].reward)
            if self.logger.episode == 0:
                self.logger.update(reward, gamelen, lastinfo['info_block'])
        else:
            if self.logger.episode == 0:
                if self.logger.supervised_test_interval is not None:
                    reward = self.run_worker(1)
                else:
                    reward = 0
                self.logger.update(reward, 0)
        for _ in progress:
            # if hasattr(self, '_train_special'):
            #     self._train_special(episode, total_rewards, total_rewards_smooth)
            if self.logger.is_supervised:
                tid = np.random.choice(self.training_memory.reward.shape[0], batch_size)
                buffer = [x[tid] for x in self.training_memory]
                buffer = self.memory.tuple(*buffer)
                trainingbuffer = self.run_episode_outputlayer(buffer, 'all')
            else:
                buffer = self.memory.sample(batch_size)
                trainingbuffer = self.run_episode_outputlayer(buffer, 'actionvalue')
            self.optimizer.zero_grad()
            loss, info_loss = self.loss.loss(buffer, trainingbuffer)
            loss.backward()
            if self.gradientclipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradientclipping)
                
            self.optimizer.step()
            if tqdmstr is None:
                tqdmstr = f"Process {tqdmpos}"
            
            progress.set_description(f"{tqdmstr}, {self.logger.getdescription()}, Loss: {loss.item():.4f}")

            if not self.logger.is_supervised:
                if not is_online:
                    reward, lastinfo = self.run_worker(batch_size)
                else:
                    reward, lastinfo = self.run_worker(1)
                gamelen = len(self.memory.memory[-1].reward)
                self.logger.update(reward, gamelen, lastinfo['info_block'])
                self.logger.save(self.model.state_dict())
            else:
                if self.logger.supervised_test_interval is not None and self.logger.episode % self.logger.supervised_test_interval == 0:
                    reward = self.run_worker(1)
                else:
                    reward = 0
                self.logger.update(reward, info_loss['perror'])
                self.logger.save(self.model.state_dict())

    def load_memory(self, filename):
        with open(filename, "rb") as f:
            (obs, action, reward, timestep, done) = pickle.load(f)
        self.training_memory = self.memory.tuple(obs, action, reward, timestep, done)
        
