
import torch
from tqdm import tqdm 
import pandas as pd

    def record(self, model, savename, n_episode = 1, showprogress = True, *arg, **kwarg):
        rg = range(n_episode)
        if showprogress:
            rg = tqdm(rg)
        recordings = []
        behaviors = []
        for i in rg:
            behavior, recording = self.run_episode(model, record = True, *arg, **kwarg)
            behaviors.append(behavior)
            recordings.append(recording)
        behaviors = pd.concat(behaviors)
        recordings = torch.concat(recordings).numpy()
        recordings = pd.DataFrame(recordings)
        if savename is not None:
            behaviors.to_csv(savename)
            recordings.to_csv(savename.replace('data_', 'recordings_'))
        if behaviors.shape[0] == recordings.shape[0]:
            print(f"recording complete: format check passed.")
        return behaviors, recordings

    def run_episode(self, model, device = "cpu", mode_action = "softmax", record = False):
        assert model is not None
        if hasattr(model, 'eval'):
            model.eval()
        done = False
        total_reward = 0
        env = self.env
        env.saveon()
        obs = env.reset(reset_task = False, clear_data = True)
        mem_state = None
        if record:
            recording_neurons = []
        while not done:
            # take actions
            obs = torch.from_numpy(obs).unsqueeze(0).float()
            action_vector, val_estimate, mem_state = model(obs.unsqueeze(0).to(device), mem_state)
            if record:
                recording_neurons.append(mem_state[0])
            action = self.select_action(action_vector, mode_action)
            obs_new, reward, done, timestep = env.step(action.item())
            reward = float(reward)
            obs = obs_new
            total_reward += reward
        if record:
            recording_neurons = torch.concat(recording_neurons).squeeze()
        return env._data, recording_neurons if record else total_reward

    def select_action(self, action_vector, mode_action):
        if mode_action == "softmax":
            action_dist = torch.nn.functional.softmax(action_vector, dim = -1)
            action_cat = torch.distributions.Categorical(action_dist.squeeze())
            action = action_cat.sample()
        return action