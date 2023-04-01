from W_RNN.W_Worker import W_Worker
from W_RNN.W_Logger import W_Logger 
import torch
import numpy as np
from tqdm import tqdm 
# from parallelbar import progress_map as tqdm
from W_Python import W_tools as W

class W_loss:
    loss_name = None
    def __init__(self, loss, device):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"enabling {self.device}")
        else:
            self.device = device
        
        self.loss_name = loss['name']
        self.params = loss['params']

    def loss(self, *arg, **kwarg):
        if self.loss_name == "A2C":
            return self.loss_A2C(*arg, **kwarg)        

    def loss_A2C(self, buffer, trainingbuffer):
        gamma = self.params['gamma']
        (_, _, reward, _, done) = buffer
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
        return loss

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

    def set_optim(self, param_optim):
        params = list(self.model.parameters())
        # params += list(self.model.init0)
        if param_optim['name'] == "RMSprop":
            self.optimizer = torch.optim.RMSprop(params, lr = param_optim['lr'])
    
    def loadlastdict(self, loadname, isresume = False):
        if loadname is None:
            return
        modeldata = torch.load(loadname)
        self.model.load_state_dict(modeldata['state_dict'])
        if isresume:
            self.logger.setlog(modeldata['info'])

    def train(self, max_episodes, batch_size, is_online = True, tqdmpos = 0):
        self.logger.init(max_episodes)
        progress = tqdm(range(0, max_episodes), position = tqdmpos, leave=True)
        self.progressbar = progress
        reward = self.run_worker(batch_size)
        gamelen = len(self.memory.memory[-1].reward)
        self.logger.update(reward, gamelen)
        for episode in progress:
            # if hasattr(self, '_train_special'):
            #     self._train_special(episode, total_rewards, total_rewards_smooth)
            # W.W_tic()
            buffer = self.memory.sample(batch_size)
            trainingbuffer = self.run_episode_outputlayer(buffer)
            self.optimizer.zero_grad()
            loss = self.loss.loss(buffer, trainingbuffer)
            loss.backward()
            if self.gradientclipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradientclipping)
                
            self.optimizer.step()
            # W.W_toc("update time = ")
            self.logger.save(self.model.state_dict())

            progress.set_description(f"Process {tqdmpos}, {self.logger.getdescription()}")
            # progress.update()
            if not is_online:
                reward = self.run_worker(batch_size)
            else:
                reward = self.run_worker(1)
            gamelen = len(self.memory.memory[-1].reward)
            self.logger.update(reward, gamelen)
