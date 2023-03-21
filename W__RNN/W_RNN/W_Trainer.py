from W_Worker import W_Worker, Data_Episode_Tuple_Train
import torch
import numpy as np
from tqdm import tqdm 

class W_loss:
    loss_name = None
    def __init__(self, loss):
        self.loss_name = loss['name']
        self.params = loss['params']

    def loss(self, *arg, **kwarg):
        if self.loss_name == "A2C":
            return self.loss_A2C(*arg, **kwarg)        

    def loss_A2C(self, buffer):
        gamma = self.params['gamma']
        # bootstrap discounted returns with final value estimates
        nstep = len(buffer)
        returns = 0
        advantages = 0
        last_value = 0
        all_returns = torch.zeros(nstep)
        all_advantages = torch.zeros(nstep)
        # run Generalized Advantage Estimation, calculate returns, advantages
        for t in reversed(range(nstep)):
            # ('state', 'action', 'reward', 'timestep', 'done', 'action_dist', 'value')
            _, _, reward, _, done, _, value = buffer[t]
            mask = 1 - done
            reward = float(reward)
            returns = reward + returns * gamma * mask
            deltas = reward + last_value * gamma * mask - value.data
            
            advantages = advantages * gamma * mask + deltas

            all_returns[t] = returns 
            all_advantages[t] = advantages
            last_value = value.data

        batch = Data_Episode_Tuple_Train(*zip(*buffer))

        action_prob = torch.cat(batch.action_prob, dim = 1).squeeze()
        action = torch.cat(batch.action, dim = 0)
        values = torch.tensor(batch.value)

        logits = (action_prob * action).sum(1)
        policy_loss = -(torch.log(logits) * all_advantages).mean()
        value_loss = 0.5 * (all_returns - values).pow(2).mean()
        entropy_reg = -(action_prob * torch.log(action_prob)).mean()

        loss = self.params['coef_valueloss'] * value_loss + policy_loss - self.params['coef_entropyloss'] * entropy_reg

        return loss 


class W_Trainer(W_Worker): 
    # env
    # model
    # loss
    # logger
    # optimizer
    # device (for training)
    def __init__(self, env, model, param_loss, param_optim, logger = None, gradientclipping = None):
        super().__init__(env, model, mode = "train")
        self.loss = W_loss(param_loss)
        self.set_optim(param_optim)
        self.logger = logger
        self.gradientclipping = gradientclipping

    def set_optim(self, param_optim):
        if param_optim['name'] == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = param_optim['lr'])

    def train(self, max_episodes, save_path = None, save_interval = 1000):
        if save_path is not None:
            save_path = save_path + "_{epi:04d}"
        total_rewards = np.zeros(max_episodes)
        progress = tqdm(range(0, max_episodes))
        for episode in progress:
            reward = self.run_episode()
            buffer = self.memory.get_last()
            self.optimizer.zero_grad()
            loss = self.loss.loss(buffer)
            loss.backward()
            if self.gradientclipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradientclipping)
            self.optimizer.step()
            total_rewards[episode] = reward

            avg_reward_100 = total_rewards[max(0, episode-100):(episode+1)].mean()

            if save_path is not None and (episode+1) % save_interval == 0:
                torch.save({
                    "state_dict": self.model.state_dict(),
                    "avg_reward_100": avg_reward_100,
                    'last_episode': episode,
                }, save_path.format(epi=episode+1) + ".pt")

            progress.set_description(f"Episode {episode+1}/{max_episodes} | \
                                      Reward: {reward} | mean Reward: {avg_reward_100:.4f} | Loss: {loss.item():.4f}")
            