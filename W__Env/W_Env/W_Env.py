import pygame
from gym.wrappers import RecordVideo
import numpy as np
import torch
from tqdm import tqdm 
import pandas as pd

def W_Env_creater(envname, *arg, **kwarg):
    tsk_name = "task_" + envname
    ldic = locals()
    exec(f"from W_Env.{tsk_name} import {tsk_name} as W_tsk", globals(), ldic)
    W_tsk = ldic['W_tsk']
    env = W_tsk(*arg, **kwarg)
    return env

class W_Env():
    env = None
    envmode = None
    envname = None
    player = None
    def __init__(self, envname, mode_env = None, is_record = False, log_dir = './video', \
                 *arg, **kwarg):
        env = W_Env_creater(envname, *arg, **kwarg)
        if is_record:
            env = self.wrap_env(env, log_dir)
        self.env = env
        self.envname = envname
        self.envmode = mode_env
        if self.envmode == "oracle_human":
            self.env.set_human_keys(['space'], [0])

    def wrap_env(self, env, log_dir):
        env = RecordVideo(env, log_dir)
        return env
    
    def get_env(self):
        return self.env
    
    def _get_human_keypress(self, mode = "human"):
        assert self.env.human_key_action is not None
        action = None
        while action is None:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    key = pygame.key.name(event.key)
                    if key == "escape":
                        self.env.render_close()
                        return None
                    elif key == "space" and mode == "oracle_human":
                        action = self.env.get_oracle_action() 
                    elif key in self.env.human_key_action['keys']:
                        kid = np.where([key == i for i in self.env.human_key_action['keys']])[0][0]
                        action = self.env.human_key_action['actions'][kid]
        return action
    
    def _get_action(self, mode, model):
        if mode == "model":
            action = model.predict(obs)
        elif mode == "random":
            action = self.env.action_space.sample()
        elif mode in ["human", "oracle_human"]:
            action = self._get_human_keypress()
        return action
    
    def play(self, mode = None, model = None):
        env = self.env
        obs = env.reset()
        done = False
        while not done:
            action = self._get_action(mode, model)
            if action is None:
                return
            obs, reward, done, timemark = env.step(action)    

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