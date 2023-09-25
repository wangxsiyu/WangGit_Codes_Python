import numpy as np
import torch
from tqdm import tqdm 
import pandas as pd
import os
import re
from W_Python.W import W

class W_Worker:
    env = None
    model = None
    mode_action = None
    device = None
    def __init__(self, env, model, device = "cpu", mode_action = "softmax", *arg, **kwarg):
        self.device = device
        self.reload_env(env)
        self.model = model.to(device)
        self.mode_action = mode_action

    def find_latest_model(self, currentfolder):
        file_trained_list = os.listdir(currentfolder)
        fs = [re.search("(.*)_(.*).pt", x) for x in file_trained_list]
        fs = [x for x in fs if x is not None]
        if not len(fs) == 0:
            its = [x.group(2) for x in fs]
            tid = np.argmax([int(x) for x in its])
            filename = os.path.join(currentfolder, fs[tid].group(0))
            iterid = int(W.W_str_select_between_patterns(filename, '_', '\.',-1,-1))
        else:
            filename = None
            iterid = 0
        return filename, iterid
    
    def load_model(self, filename):
        if filename is not None:
            modeldata = torch.load(filename)
            print(f"loaded model {filename}")
            self.model.load_state_dict(modeldata['state_dict'])
            traininginfo = modeldata['training_info']
        else:
            traininginfo = None
        return traininginfo
    
    def reload_env(self, env):
        self.env = env

    def select_action(self, action_vector):
        if self.mode_action == "softmax":
            # q = action_vector.numpy()
            if torch.is_tensor(action_vector):
                prob = torch.nn.functional.softmax(action_vector, dim = 1)
                dist=torch.distributions.categorical.Categorical(probs=prob)
                action = dist.sample()
            else:
                prob = np.exp(action_vector) / np.sum(np.exp(action_vector))
                action = np.random.choice(np.arange(len(prob)), size = 1, p = prob) # this should be in torch
        return action
    
    def work(self, n_episode = 1, showprogress = False, *arg, **kwarg):
        rg = range(n_episode)
        if showprogress:
            rg = tqdm(rg)
        data = []
        reward = []
        for i in rg:
            treward, tdata = self.run_episode(*arg, **kwarg)
            reward.append(treward)
            data.append(tdata)
        return np.mean(reward), data
    
    def record(self, savename, n_episode = 1, is_record = True, showprogress = True, *arg, **kwarg):
        rg = range(n_episode)
        if showprogress:
            rg = tqdm(rg)
        behaviors = []
        if is_record:
            recordings = []
        for i in rg:
            if is_record:
                behavior, recording = self.run_episode(recording_mode = "neurons", *arg, **kwarg)
                recordings.append(recording)
            else:
                behavior = self.run_episode(recording_mode = "behavior", *arg, **kwarg)
            behaviors.append(behavior)
        behaviors = pd.concat(behaviors)
        if is_record:
            recordings = torch.concat(recordings).numpy()
            recordings = pd.DataFrame(recordings)
        if savename is not None:
            behaviors.to_csv(f"behavior_{savename}")
            if is_record:
                recordings.to_csv(f"recording_{savename}")
        if (not is_record) or (behaviors.shape[0] == recordings.shape[0]):
            print(f"recording complete: format check passed.")
        else:
            Warning(f"recording ends: format check failed, please check!")
        return behaviors, recordings if is_record else behaviors
    
    def run_episode(self, recording_mode = "train", *arg, **kwarg): # recording_mode: training, behavior, neurons 
        model = self.model
        env = self.env
        assert model is not None
        assert env is not None
        if recording_mode in ["behavior", "neurons"]:
            return self.run_episode_recording(env, model, recording_mode, *arg, **kwarg)
        elif recording_mode == "train":
            return self.run_episode_train(env, model, *arg, **kwarg)

    def run_episode_recording(self, env, model, recording_mode = "behavior"):
        done = False
        env.saveon() # save behavior
        model.eval()
        obs = env.reset() # return numpy array, dim (1,)
        if hasattr(model, 'initialize_latentvariables'):
            LV = model.initialize_latentvariables()
        else:
            LV = None
        if recording_mode == "neurons":
            recording_neurons = []
        while not done:
            # take actions
            obs = torch.from_numpy(obs).unsqueeze(0).float() # [obs] nBatchsize x dim_obs (suitable for RNN)
            action_vector, LV, _ = model(obs, LV) # return action, hidden_state, additional_output can be value etc
            if recording_mode == "neurons":
                recording_neurons.append(LV.squeeze())
            action = self.select_action(action_vector)
            obs_new, _, done, _ = env.step(action)
            obs = obs_new
        if recording_mode == "neurons":
            recording_neurons = torch.concat(recording_neurons).squeeze()
        if recording_mode == "neurons":
            return env._data, recording_neurons
        else:
            return env._data
        
    def run_episode_train(self, env, model):
        done = False
        env.saveoff() # do not save behavior
        model.eval()
        obs = env.reset() # return numpy array
        obs = torch.from_numpy(obs).unsqueeze(0).float()
        if hasattr(model, 'initialize_latentvariables'):
            LV = model.initialize_latentvariables()
        else:
            LV = None
        total_reward = 0
        data = {'obs':[], 'reward':[], 'action':[], 'obs_next':[], 'timestep':[], \
                'isdone': [], 'additional_output':[]}
        while not done:
            # take actions
            data['obs'].append(obs)
            action_vector, LV, additional_output = model(obs, LV) # return action, hidden_state, additional_output can be value etc
            action = self.select_action(action_vector)
            obs_new, reward, done, timestep = env.step(action.cpu().numpy())
            obs_new = torch.from_numpy(obs_new).unsqueeze(0).float()
            data['obs_next'].append(obs_new)
            data['action'].append(action.unsqueeze(0))
            data['reward'].append(torch.tensor(float(reward)).unsqueeze(0).unsqueeze(0))
            data['timestep'].append(torch.tensor(timestep).unsqueeze(0).unsqueeze(0))
            data['isdone'].append(torch.tensor(done).unsqueeze(0).unsqueeze(0))
            # if additional_output is not None:
            data['additional_output'].append(additional_output)
            obs = obs_new
            total_reward += reward
        keys = list(data.keys())
        for x in keys:
            if all([None == i for i in data[x]]):
                # print(f"no {x}: skipped")
                data.pop(x)
            else:
                data[x] = torch.concat(data[x]).float()
        # d = [np.array(x, dtype = 'float') for x in arg]
        return total_reward, data
    