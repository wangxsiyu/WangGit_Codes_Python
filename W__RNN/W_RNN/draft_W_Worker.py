import torch
from tqdm import tqdm 
import pandas as pd

class W_Worker:
    def __init__(self, env, model, *arg, **kwarg):
        self.env = env
        self.model = model
       
    def run_episode(self, recording_mode = "training", *arg, **kwarg): # recording_mode: state-action, behavior, neurons 
        model = self.model
        env = self.env
        assert model is not None
        assert env is not None
        if recording_mode in ["behavior", "neurons"]:
            self.run_episode_recording(env, model, recording_mode, *arg, **kwarg)
        elif recording_mode == "training":
            self.run_episode_training(env, model, *arg, **kwarg)

    def run_episode_recording(self, env, model, recording_mode = "behavior", device = "cpu", mode_action = "softmax"):
        done = False
        env.saveon() # save behavior
        model.eval()
        obs = env.reset() # return numpy array
        mem_state = None
        if recording_mode == "neurons":
            recording_neurons = []
        while not done:
            # take actions
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).float().to(device) # [[[obs]]] nLayer x nBatchsize x dim_obs (suitable for RNN)
            action_vector, mem_state, _ = model(obs, mem_state) # return action, hidden_state, additional_output can be value etc
            if recording_mode == "neurons":
                recording_neurons.append(mem_state.squeeze())
            action = self.select_action(action_vector, mode_action)
            obs_new, _, done, _ = env.step(action)
            obs = obs_new
        if recording_mode == "neurons":
            recording_neurons = torch.concat(recording_neurons).squeeze()
        if recording_mode == "neurons":
            return env._data, recording_neurons
        else:
            return env._data

    def run_episode_training(self, env, model, is_stateactiononly = True, device = "cpu", mode_action = "softmax"):
        done = False
        env.saveoff()
        obs = env.reset() # return numpy array
        mem_state = None
        total_reward = 0
        if is_stateactiononly:
            data = {'obs':[], 'reward':[], 'action':[], 'obs_next':[], 'timestep':[]}
        else:
            data = {'obs':[], 'reward':[], 'action':[], 'obs_next':[], 'timestep':[], 'additional_output':[]}
        while not done:
            # take actions
            data['obs'].append(obs)
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).float().to(device) # [[[obs]]] nLayer x nBatchsize x dim_obs (suitable for RNN)
            action_vector, mem_state, additional_output = model(obs, mem_state) # return action, hidden_state, additional_output can be value etc
            action = self.select_action(action_vector, mode_action)
            obs_new, reward, done, timestep = env.step(action)
            data['obs_new'].append(obs_new)
            data['action'].append(action)
            data['reward'].append(reward)
            data['timestep'].append(timestep)
            if not is_stateactiononly:
                data['additional_output'].append(additional_output)
            reward = float(reward)
            obs = obs_new
            total_reward += reward
        return total_reward, data
    
    def record(self, savename, n_episode = 1, is_record = True, showprogress = True, *arg, **kwarg):
        rg = range(n_episode)
        if showprogress:
            rg = tqdm(rg)
        behaviors = []
        if is_record:
            recordings = []
        for i in rg:
            if is_record:
                behavior, recording = self.run_episode(self.model, recording_mode = "neurons", *arg, **kwarg)
                recordings.append(recording)
            else:
                behavior = self.run_episode(self.model, recording_mode = "behavior", *arg, **kwarg)
            behaviors.append(behavior)
        behaviors = pd.concat(behaviors)
        if is_record:
            recordings = torch.concat(recordings).numpy()
            recordings = pd.DataFrame(recordings)
        if savename is not None:
            behaviors.to_csv(f"behavior_{savename}")
            recordings.to_csv(f"recording_{savename}")
        if (not is_record) or (behaviors.shape[0] == recordings.shape[0]):
            print(f"recording complete: format check passed.")
        else:
            Warning(f"recording ends: format check failed, please check!")
        return behaviors, recordings if is_record else behaviors

    def select_action(self, action_vector, mode_action = "softmax"):
        if mode_action == "softmax":
            q = action_vector.numpy()
            prob = np.exp(q) / np.sum(np.exp(q))
            action = np.random.choice(np.arange(len(prob)), size = 1, p = prob)
        return action
    
    def work(self, n_episode = 1, showprogress = False, *arg, **kwarg):
        rg = range(n_episode)
        if showprogress:
            rg = tqdm(rg)
        for i in rg:
            self.run_episode(*arg, **kwarg)
            