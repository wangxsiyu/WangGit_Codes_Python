from gym import spaces
from W_Gym.W_Gym import W_Gym
from W_Python import W_tools as W
import numpy as np

class task_Temporal_Discounting(W_Gym):
    p_reward = None
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.observation_space = spaces.Discrete(13) # 1 + 9 + 1 + 1 + 1
        # set action space
        self.action_space = spaces.Discrete(2) # fix, release
        # set rendering dimension names
        self.setup_obs_Name2DimNumber({'fixation':0, \
                                       'image':np.arange(1,10).tolist(), 'red':10, 'purple':11, \
                                        'green':12})
        # set stages
        stage_names = ["fixation", "image", "red", \
                       "purple", "purple_overtime", "green"]
        stage_advanceuponaction = ["red", "purple"]
        self.setW_stage(stage_names = stage_names, stage_advanceuponaction = stage_advanceuponaction)
        self.effective_actions = [1]

    def _setup_render(self):
        plottypes = ["circle", "image", "square", "square", "square"]
        colors = [(255,255,255), (0,0,0), (255, 0, 0), (255, 0, 255), (0,255,0)]
        radius = [0.02, 0, 0.04, 0.04, 0.04]
        self._render_setplotparams('obs', plottypes, colors, radius)
        plottypes = ["square"]
        colors = [(255,0,0)]
        radius = [0.1]
        self._render_setplotparams('action', plottypes, colors, radius)

    def _reset_trial(self):
        image = np.random.choice(9,1)
        delay = np.floor(image/3)
        drop = image % 3
        param = {'image':image, 'delay':delay, "drop": drop} # 1,5,10 (change this)
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
            self.draw("image", W.W_onehot(self.param_trial['image'], 9))
        else:
            self.draw(self.metadata_stage['stage_names'][self.stage], 1)
        self.flip()
