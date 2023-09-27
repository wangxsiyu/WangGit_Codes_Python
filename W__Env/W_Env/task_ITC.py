from W_Gym.W_Gym import W_Gym
from W_Python.W import W
from gym import spaces
import numpy as np

class task_ITC(W_Gym):
    _param_task = {'delay': [1,1,1,5,5,5,10,10,10], 'drop': [2,4,6,2,4,6,2,4,6]}
    def __init__(self, is_ITI = False, *arg, **kwarg):
        super().__init__(is_ITI = is_ITI, *arg, **kwarg)
        self.env_name = "ITC"
        self.setup_obs_dim(12)  # 9 cues + 1 red + 1 purple + 1 green
        # set action space
        self.setup_action_dim(2) # release, hold
        # set rendering dimension names
        self.setup_obs_channel_namedict({'image':np.arange(9).tolist(), 'red':9, 'purple':10, \
                                        'green':11})
        # set stages
        state_names = ["image", "red", \
                       "purple", "purple_overtime", "green"]
        state_immediateadvance = ["red", "purple"]
        self.setup_state_parameters(state_names = state_names, state_immediateadvance = state_immediateadvance, \
                        state_timelimits= [1,1,1,1,99])

    def custom_reset_trial(self):
        image = np.random.choice(9,1).astype('int32')
        delay_vals = self._param_task['delay']
        drop_vals = self._param_task['drop']
        self._param_trial = {'image':image, 'delay':delay_vals[image], "drop": drop_vals[image]} # 1,5,10 (change this)
        self._env_vars['choice'] = None

    def custom_step_set_validactions(self):
        if self._metadata_state['statenames'][self._state] in ["image","green"]:
            self._state_valid_actions = [0]
        elif self._metadata_state['statenames'][self._state] in ["red", "purple"]:
            self._state_valid_actions = [0,1]

    def custom_step_reward(self, action):
        R = 0
        # register pos choice 1 and choice 2
        if self._env_vars['choice'] is None:
            if self._metadata_state['statenames'][self._state] in ["red"] and action == 1:
                self._env_vars['choice'] = "reject"
            elif self._metadata_state['statenames'][self._state] in ["purple"] and action == 1: 
                self._env_vars['choice'] = "accept"
        if self._metadata_state['statenames'][self._state] == "image":    
            sid = self._find_state('green')
            self._metadata_state['state_timelimits'][sid] = self._param_trial['delay']
        return R
    
    def custom_state_transition(self, action, is_effective = True):
        if self._metadata_state['statenames'][self._state] == "red":
            state = self._advance_trial()
        elif self._metadata_state['statenames'][self._state] == "purple":
            state = self._go_to_state('green')
        else:
            state = self._state + 1

        if state < len(self._metadata_state['statenames']) and self._metadata_state['statenames'][state] == "purple_overtime":
            is_error = True 
        return is_error, R, is_done

    def _step_after(self, action):
        R_ext = 0
        R_int = 0

        # get reward
        if not self.trial_is_error and self.trial_choice == "accept" and \
            self._metadata_state['statenames'][self._state] == "ITI":
            R_ext += self.param_trial['drop'] * self.Rewards['R_reward']
        return R_ext, R_int

    def _draw_obs(self):
        if self._metadata_state['statenames'][self._state] == "image":
            timg = W.W_onehot(self.param_trial['image'], 9)
            self.draw("image", timg)
        else:
            self.draw(self._metadata_state['statenames'][self._state], 1)
        self.flip()

    def _render_frame_obs_format(self, obs, lst):
        c = 0
        for i, j in lst.items():
            if i == "image":
                obs[c] = obs[c].reshape((3,3))
                obs[c] = obs[c] * 128 + np.any(obs[c] > 0) * 127
            c += 1
        return obs
    
    def setup_render_parameters(self):
        plottypes = ["circle", "image", "square", "square", "square"]
        colors = [(255,255,255), (0,0,0), (255, 0, 0), (255, 0, 255), (0,255,0)]
        radius = [0.02, 0.1, 0.04, 0.04, 0.04]
        self._render_set_auto_parameters('obs', plottypes, colors, radius)
        plottypes = ["arrows"]
        plotparams = [1,0,2,-1]
        self._render_set_auto_parameters('action', plottypes, plotparams = plotparams)
    
