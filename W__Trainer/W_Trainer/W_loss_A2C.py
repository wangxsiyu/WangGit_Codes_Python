 
import torch

class W_loss_A2C:
    def loss_A2C(self, buffer, trainingbuffer):
        gamma = self.loss_params['gamma']        
        reward = buffer.reward.squeeze(-1).to(self.device)
        done = buffer.isdone.squeeze(-1).to(self.device)
        # bootstrap discounted returns with final value estimates
        value = trainingbuffer.outputs.squeeze(-1)
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

        logll = torch.log(trainingbuffer.action_likelihood)
        policy_loss = -(logll * all_advantages).mean()
        value_loss = 0.5 * (all_returns - value).pow(2).mean()

        action_dist = trainingbuffer.action_dist
        entropy_reg = -(action_dist * torch.log(action_dist)).mean()

        loss_actor = policy_loss - self.loss_params['coef_entropyloss'] * entropy_reg
        loss_critic = self.loss_params['coef_valueloss'] * value_loss  
        loss = loss_actor + loss_critic

        if torch.isinf(loss):
            print('check: infinite loss')
        return loss, None
    