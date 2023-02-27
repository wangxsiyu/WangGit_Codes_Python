from gym import spaces
from W_Gym.W_Gym import W_Gym
from W_Python import W_tools as W
import numpy as np

class task_Horizon(W_Gym):
    task_param = {'mu':[40, 60], 'sd':8, 'diff': [-20,-12,-8,-4,4,8,12,20], \
        'n_instructed':4, 'horizon':[1,6]}
    def __init__(self, *arg, **kwarg):
        super().__init__(is_augment_obs = False, *arg, **kwarg)
        self.observation_space = spaces.Discrete(15) # 1 + 2 + 2 + 10
        # set action space
        self.action_space = spaces.Discrete(3) # fix, L, R
        # set rendering dimension names
        self.setup_obs_Name2DimNumber({'fixation':0, \
                                       'cue':[1,2], \
                                        'reward':[3,4], \
                                        'horizon': np.arange(5, 15).tolist()})
        # set stages
        stage_names = ["fixation", "horizon", "choice", "reward"]
        stage_advanceuponaction = ["choice"]
        self.setW_stage(stage_names = stage_names, stage_advanceuponaction = stage_advanceuponaction)
        self.effective_actions = [1, 2]

    def _setup_render(self):
        plottypes = ["circle", "image", "square", "square", "square"]
        colors = [(255,255,255), (0,0,0), (255, 0, 0), (255, 0, 255), (0,255,0)]
        radius = [0.02, 0.1, 0.04, 0.04, 0.04]
        self._render_setplotparams('obs', plottypes, colors, radius)
        plottypes = ["square"]
        colors = [(255,0,0)]
        radius = [0.1]
        self._render_setplotparams('action', plottypes, colors, radius)

    def _reset_trial(self):
        image = np.random.choice(9,1)
        delay_vals = np.array([1, 5, 10])
        drop_vals = np.array([2, 4, 6])
        delay = np.floor(image/3)
        drop = image % 3
        param = {'image':image, 'delay':delay_vals[delay.astype('int32')], "drop": drop_vals[drop]} # 1,5,10 (change this)
        self.param_trial = param
        self.trial_choice = None
    
    def _step(self, action):
        R_ext = 0
        R_int = 0
        # register pos choice 1 and choice 2
        if self.trial_choice is None:
            if self.metadata_stage['stage_names'][self.stage] in ["red"] and action != 0:
                self.trial_choice = "reject"
            elif self.metadata_stage['stage_names'][self.stage] in ["purple"] and action != 0: 
                self.trial_choice = "accept"
        if self.metadata_stage['stage_names'][self.stage] == "image":    
            sid = self.find_stage('green')
            self.metadata_stage['stage_timings'][sid] = self.param_trial['delay']
        return R_ext, R_int
    
    def _advance_stage(self):
        is_error = False
        if self.metadata_stage['stage_names'][self.stage] == "red" and self.is_effective_action:
            stage = self.find_stage('ITI')
        elif self.metadata_stage['stage_names'][self.stage] == "purple" and self.is_effective_action:
            stage = self.find_stage('green')
        else:
            stage = self.stage + 1

        if stage < len(self.metadata_stage['stage_names']) and self.metadata_stage['stage_names'][stage] == "purple_overtime":
            is_error = True 
        return stage, is_error

    def _step_after(self, action):
        R_ext = 0
        R_int = 0

        # get reward
        if not self.trial_is_error and self.trial_choice == "accept" and \
            self.metadata_stage['stage_names'][self.stage] == "ITI":
            R_ext += self.param_trial['drop'] * self.Rewards['R_reward']
        return R_ext, R_int

    def _draw_obs(self):
        if self.metadata_stage['stage_names'][self.stage] == "image":
            timg = W.W_onehot(self.param_trial['image'], 9)
            self.draw("image", timg)
        else:
            self.draw(self.metadata_stage['stage_names'][self.stage], 1)
        self.flip()

    def _render_frame_obs_format(self, obs, lst):
        c = 0
        for i, j in lst.items():
            if i == "image":
                obs[c] = obs[c].reshape((3,3))
                obs[c] = obs[c] * 128 + np.any(obs[c] > 0) * 127
            c += 1
        return obs
