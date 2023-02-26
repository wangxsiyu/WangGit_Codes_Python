from W_Python import W_tools as W
import gym
import numpy as np

class W_Gym_render(gym.Env):
    dt = None
    currentscreen = None
    window = None
    clock = None
    metadata_render = {'render_mode': None, 'window_size': [512, 512], 'render_fps': None}
    def __init__(self, render_mode = None, window_size = [512, 512], \
                 render_fps = None, dt = 1, \
                 **kwarg):
        self.dt = dt
        metadata_render = W.W_dict_kwargs()
        W.W_dict_updateonly(self.metadata_render, metadata_render)
        if self.metadata_render['render_fps'] is None:
            self.metadata_render['render_fps'] = 1/self.dt
        self.metadata_render['window_size'] = np.array(self.metadata_render['window_size'])
        self.setup_rendermode()
        if hasattr(self, '_setup_render'):
            self._setup_render()
        print(f"render mode: {self.metadata_render['render_mode']}")

    def flip(self, is_clear = True):
        self.obs = self.currentscreen
        if is_clear:
            self.blankscreen()

    def blankscreen(self):
        assert hasattr(self, 'observation_space')
        self.currentscreen = np.zeros(self.observation_space.shape)

    def setup_rendermode(self, render_mode = None):
        if render_mode is None:
            render_mode = self.metadata_render['render_mode']
        if render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.metadata_render['window_size'])
            self.clock = pygame.time.Clock()

    def _render_frame_update(self, canvas):
        if self.metadata_render['render_mode'] == "human":
            import pygame
            assert self.window is not None
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata_render["render_fps"])
        else:  # rgb_array or single_rgb_array
            import numpy as np
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def _render_frame_create(self):
        import pygame 
        canvas = pygame.Surface(self.metadata_render['window_size'])
        canvas.fill((0, 0, 0))
        return canvas
    
    def close(self):
        if self.window is not None:
            import pygame 
            pygame.display.quit()
            pygame.quit()

    def render(self, option = None, *arg, **kwarg):
        if self.metadata_render['render_mode'] is None:
            return
        canvas = self._render_frame_create()
        option = W.enlist(option)
        for x in option:
            canvas = self.render_frame(canvas, x, *arg, **kwarg)
        self._render_frame_update(canvas)

    def render_frame(self, canvas, option = None, *arg, **kwarg):
        if option is None:
            assert hasattr(self, '_render_frame')
            return self._render_frame(canvas, *arg, **kwarg)    
        elif option == "action":
            return self._render_frame_action(canvas, *arg, **kwarg)
        elif option == "obs":
            return self._render_frame_obs(canvas, *arg, **kwarg)
        elif option == "reward":
            return self._render_frame_reward(canvas, *arg, **kwarg)
        else:
            assert hasattr(self, '_render_frame')
            return self._render_frame(canvas, option, *arg, **kwarg)

    def _render_frame_obs(self, canvas, *arg, **kwarg):
        return canvas
    
    def _render_frame_reward(self, canvas, *arg, **kwarg):
        return canvas

    def _render_frame_action(self, canvas, *arg, **kwarg):
        return canvas

    def get_window_relative2absolute(self, pos):
        import numpy as np
        pos = np.array(pos)
        return pos * self.metadata_render['window_size']

class W_Gym(W_Gym_render):
    Rewards = {"R_advance":0, "R_error": -1, "R_reward": 1}
    obs = None
    obs_augment = None
    is_augment_obs = False
    is_faltten_obs = True
    info = None
    n_maxTrials = None
    t = 0 # total time from beginning
    timer = 0 # timer
    stage = 0 # task stage
    tot_trials = 0 # total trials from beginning
    trial_counter = 0 # can be reset
    metadata_episode = {"n_maxTrialsPerBlock": np.Inf, "n_maxBlocks": np.Inf}
    metadata_stage = {'stage_names':None, 'stage_timings': None, 'stage_advanceuponaction': None}
    last_reward = 0
    last_action = None
    def __init__(self, n_maxTrials = np.Inf, \
                 n_maxTrialsPerBlock = np.Inf, n_maxBlocks = np.Inf, \
                 is_augment_obs = False, is_faltten_obs = True, **kwarg):
        super().__init__(**kwarg)
        self.n_maxTrials = n_maxTrials
        self.is_augment_obs = is_augment_obs
        self.is_faltten_obs = is_faltten_obs or self.is_augment_obs
        metadata = W.W_dict_kwargs()
        W.W_dict_updateonly(self.metadata_episode, metadata)
        self.setW_stage(["stages"], [np.Inf])

    # flow
    def reset(self, return_info = False):
        self.tot_trials = 0
        if hasattr(self, '_reset'):
            self._reset()
        self.reset_block()
        self.reset_trial()
        self.render(option = 'obs')
        obs = self._get_obs()
        info = self._get_info()
        return obs if not return_info else (obs, info)

    def reset_block(self):
        self.trial_counter = 0
        if hasattr(self, '_reset_block'):
            self._reset_block()

    def reset_trial(self):
        self.t = 0
        self.timer = 0
        self.stage = 0
        self.valid_choices = None
        if hasattr(self, '_reset_trial'):
            self._reset_trial()
    
    def step(self, action):
        self.t = self.t + self.dt
        self.timer = self.timer + self.dt
        reward_E, reward_I, is_done = self._step(action)
        self.render(option = ["obs","action","reward"])
        self.last_reward = reward_E + reward_I
        obs = self._get_obs()
        info = self._get_info()
        return obs, self.last_reward, is_done, None, info
    
    def advance_stage(self, is_error = False):
        is_done = False
        R_ext = 0
        is_nexttrial = 0
        if is_error:
            R_int = self.Rewards['R_error']
            is_nexttrial = 1
        else:
            R_int = self.Rewards['R_advance']
            self.stage = self.stage + 1
            self.timer = 0 # auto reset timer 
            if self.stage == len(self.metadata_stage['stage_names']):
                is_nexttrial = 1
        
        if is_nexttrial == 1:
            self.tot_trials += 1
            self.trial_counter += 1
            if self.tot_trials >= self.n_maxTrials:
                is_done = True
            if self.trial_counter >= self.metadata_episode['n_maxTrialsPerBlock']:
                self.reset_block()
            self.reset_trial()   

        return R_ext, R_int, is_done

    # rewards
    def setW_reward(self, **kwarg):
        W.W_dict_updateonly(self.Rewards, kwarg)

    def setW_stage(self, stage_names, stage_timings = None, \
                   stage_advanceuponaction = None):
        self.metadata_stage['stage_names'] = stage_names
        nstage = len(self.metadata_stage['stage_names'])
        if stage_timings is None:
            stage_timings = np.ones(nstage) * self.dt    
        if stage_advanceuponaction is None:
            stage_advanceuponaction = np.zeros(nstage)
        else:
            c = np.zeros(nstage)
            tid = np.array([np.where([j == i for j in self.metadata_stage['stage_names']]) for i in iter(stage_advanceuponaction)]).squeeze()
            c[tid] = 1
            stage_advanceuponaction = c
        self.metadata_stage['stage_timings'] = stage_timings
        self.metadata_stage['stage_advanceuponaction'] = stage_advanceuponaction

    def probabilistic_reward(self, p):
        if np.random.uniform() <= p:
            reward = self.Rewards['R_reward']
        else:
            reward = 0
        return reward
    # action
    def check_isvalidaction(self, action, valid_actions = None):
        if valid_actions is None:
            return True
        else:
            valid_actions = W.enlist(valid_actions)
            is_valid = action in valid_actions
            return is_valid

    # get obs
    def _get_obs(self):
        if self.is_faltten_obs or self.is_augment_obs:
            obs = self.obs.flatten()
        else:
            obs = self.obs
        if self.is_augment_obs:
            action = np.zeros(self.action_space.n)
            if self.last_action is not None:
                action[self.last_action] = 1
            obs = np.concatenate((obs, [self.last_reward], action))
        return obs
    
    def _get_info(self):
        return self.info