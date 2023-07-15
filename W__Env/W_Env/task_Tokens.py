from gym import spaces
from W_Gym.W_Gym_Grid2D import W_Gym_grid2D
from W_Python.W import W
import numpy as np
import random

class task_Tokens(W_Gym_grid2D):
    _param_task = {'p_reward': 1, 'CueValues': np.array([1,2,-1,-2]), 'endowment': 0, \
                   }
    def __init__(self, **kwarg):
        super().__init__(1,3,5, is_ITI = False, n_maxTrialsPerBlock = 180, **kwarg)
        # set action space
        self.action_space = spaces.Discrete(3) # stay, left, right
        # set rendering dimension names
        self.setup_obs_channel_namedict({'fixation':0, 'image1':1, 'image2':2, 'image3':3, 'image4':4})
        # set stages
        state_names = ["fixation", \
                       "choice"]
        state_immediateadvance = ["choice"]
        self.setup_state_parameters(state_names = state_names, state_immediateadvance = state_immediateadvance,\
                        state_timelimits = "ones")        
        self.setup_human_keys_auto('binary_plus')

    def custom_reset_block(self):
        self._param_block['ImageValues'] = random.shuffle(self._param_task['CueValues'])

    def custom_reset_trial(self):
        self.gaze.set_pos(0,1)
        IM_id = np.random.choice(4,2, replace = False)
        R_side = [self.image_R[x] for x in IM_id]
        randv = 0+(np.random.rand(2) < self.p_reward)
        R_side = R_side * randv
        self._param_trial = {'IM_id': IM_id, 'R_side': R_side}

    def _step_set_validactions(self):
        if self.metadata_stage['stage_names'][self.stage] in ["fixation"]:
            self.valid_actions = 1
        elif self.metadata_stage['stage_names'][self.stage] == "choice":
            self.valid_actions = [0,2]
    
    def _draw_obs(self):
        # calculate new observation
        if self.metadata_stage['stage_names'][self.stage] in ["fixation"]:
            self.draw((0,1),"fixation")
        elif self.metadata_stage['stage_names'][self.stage] == "choice":
            for i in range(2):
                IMid = self.param_trial['IM_id'][i]
                tstr = "image" + str(IMid+1)
                self.draw((0, i*2), tstr)
        self.flip()


    def _step(self, action):
        R_ext = 0
        R_int = 0
        # register pos choice 1 and choice 2
        if self.metadata_stage['stage_names'][self.stage] == "choice":
            self.choice = int(action/2)      
            treward = self.param_trial['R_side'][self.choice]
            R_ext += treward     
        return R_ext, R_int

    def _setup_render(self):
        plottypes = ["circle", "square", "square", "square", "square"]
        colors = [(255,255,255), (0,100,100), (0, 255, 0), (0, 0, 255), (100,100,0)]
        radius = [0.02, 0.08, 0.08, 0.08, 0.08]
        self._render_setplotparams('obs', plottypes, colors, radius)
        plottypes = ["circle"]
        colors = [(255,0,0)]
        radius = [0.01]
        self._render_setplotparams('action', plottypes, colors, radius)
        
        

    