# from parallelbar import progress_map as tqdm
from W_Python import W_tools as W
import os
import re

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"enabling {self.device}")


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
 

    


        
