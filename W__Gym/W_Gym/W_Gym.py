from W_Python import W_tools as W
import numpy as np

class W_Gym():
    """
    It implements a class of episodic task consists of blocks, trials, states and steps
    """
    # metadata - default task parameters
    _param_task = {"n_maxT": np.Inf, "n_maxTrials": np.Inf, "n_maxBlocks": np.Inf}
    _param_block =  {"block_n_maxTrials": np.Inf}
    _param_trial = {}
    _param_state = {}

    # information about task settings
    _info_task = None
    _info_block = None
    _info_trial = None
    # timing variables (unit time_unit ms)
    _time_unit = 1
    _time_task = 0
    _time_block = 0
    _time_trial = 0
    _time_state = 0
    # counting variables (number of steps)
    _count_block = 0
    _count_trial = 0
    # default variables
    _obs = None # observation
    # stage variables
    _state = 0
    _state_valid_actions = None
    _state_effective_actions = None
    # data variable (recorded behavior)
    _data_task = None
    _data_block = None
    _data_trial = None
    _data_state = None
    
    def __init__(self):
        metadata_render = W.W_dict_kwargs()
        W.W_dict_updateonly(self.metadata_render, metadata_render)
    
    def reset(self, return_info = False):
        self._data_task = None
        self._time_task = 0
        self._count_block = 0
        if hasattr(self, 'custom_reset'):
            self.custom_reset()
        self._reset_block()

        obs = self._get_obs()
        info = self._get_info()
        return obs if not return_info else (obs, info)

    def _reset_block(self):
        self._data_block = None
        self._count_block += 1
        self._count_trial = 0
        self._time_block = 0
        if hasattr(self, 'custom_reset_block'):
            self.custom_reset_block()
        self._reset_trial()

    def _reset_trial(self):
        self._data_trial = None
        self._count_trial += 1
        self._time_trial = 0
        if hasattr(self, 'custom_reset_trial'):
            self.custom_reset_trial()
        self._state = 0
        self._reset_state()

    def _reset_state(self):
        self._data_state = None
        self._time_state = 0
        self._state_valid_actions = None
        self._state_effective_actions = None

    def step(self, action):
        reward = 0
        # advance time
        self._advance_time()
        # transform actions
        if hasattr(self, "transform_actions"):
            action = self.transform_actions(action)
        # check valid actions
        is_error = not self._check_validactions(action, self._state_valid_actions)
        is_effective = self._check_validactions(action, self._state_effective_actions)
        # record current action
        self._record('action', action)
        # get consequences of actions (reward)
        if not is_error and hasattr(self, 'custom_step_reward'):
            treward = self.custom_step_reward(action)
            reward += treward
        # get consequences of actions (state transition)
        


    def _advance_time(self):
        self._time_state += self._time_unit
        self._time_trial += self._time_unit
        self._time_block += self._time_unit
        self._time_task += self._time_unit

    def _check_validactions(self, action, valid_actions = None):
        if valid_actions is None:
            return True
        else:
            valid_actions = W.enlist(valid_actions)
            is_valid = action in valid_actions
            return is_valid
        
    def _record(self, varname, var):
        pass

        

        