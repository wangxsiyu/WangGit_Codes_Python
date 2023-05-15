import pygame
from gym.wrappers import RecordVideo
import numpy as np

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

    
