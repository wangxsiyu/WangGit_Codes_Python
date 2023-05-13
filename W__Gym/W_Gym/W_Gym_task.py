from W_Python import W_tools as W
import numpy as np
import gym

class W_Gym_task(gym.Env):
    """
    It implements a class of episodic task consists of blocks, trials, states and steps
    """
    # gym parameters
    _param_gym = {"n_maxTime": np.Inf, "n_maxTrials": np.Inf, "n_maxBlocks": np.Inf, \
                    "block_n_maxTrials": np.Inf, \
                    "option_augment_obs": ["action", "reward"], \
                    "is_flatten_obs": True, "is_ITI": True}
    # reward setting
    _param_rewards = {"R_error": -1, "R_reward": 1}
    # parameters for task, block and trial
    _param_task = None
    _param_block = None
    _param_trial = None
    _param_state = {'n': 0, 'names': None, 'timelimits': None, \
                    'immediate_advance': None, 'transition_matrix': None} 
    # timing variables (unit ms)
    _time_unit = 1 # ms
    _time_task = 0 # time since task start 
    _time_block = 0 # time since block start
    _time_trial = 0 # time since trial start
    _time_state = 0 # time since state start
    # counting variables (number of steps)
    _count_block = 0 # number of blocks completed
    _count_block_trial = 0 # number of trials completed since block start
    _count_task_trial = 0 # number of trials completed since task start
    # gym variables
    _obs = None # observation
    # trial variables
    _trial_is_error = False
    # state variables
    _state = 0 # current state
    _state_valid_actions = None # valid actions for a state
    _state_effective_actions = None # effective actions for a state
    # data variable (recorded behavior)
    _data_task = None # data for task 
    _data_block = None # data for block
    _data_trial = None # data for trial
    _data_state = None # data for state
    
    def __init__(self, n_maxTime = np.Inf, n_maxTrials = np.Inf, n_maxBlocks = np.Inf, \
                    block_n_maxTrials = np.Inf, option_augment_obs = ["action", "reward"], \
                    is_flatten_obs = True, is_ITI = True, dt = 1, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        _param_inputs = W.W_dict_kwargs()
        W.W_dict_updateonly(self._param_gym, _param_inputs)
        if self._param_gym['option_augment_obs'] is not None:
            self._param_gym['is_flatten_obs'] = True
        self._time_unit = dt

    def setup_reward(self, **kwarg):
        W.W_dict_updateonly(self._param_rewards, kwarg)

    def reset(self):
        self._param_task = None
        self._data_task = None
        self._time_task = 0
        self._count_block = 0
        self._task_trial = 0
        if hasattr(self, 'custom_reset'):
            self.custom_reset()
        self._reset_block()
        obs = self._get_obs()
        return obs
    
    def _reset_block(self):
        self._param_block = None
        self._data_block = None
        self._count_block_trial = 0
        self._time_block = 0
        if hasattr(self, 'custom_reset_block'):
            self.custom_reset_block()
        self._reset_trial()

    def _reset_trial(self):
        self._param_trial = None
        self._data_trial = None
        self._time_trial = 0
        self._trial_is_error = False
        if hasattr(self, 'custom_reset_trial'):
            self.custom_reset_trial()
        self._state = 0
        self._reset_state()
        
    def _reset_state(self):
        self._data_state = None
        self._time_state = 0
        self._state_valid_actions = None
        self._state_effective_actions = None
        # set valid actions for the new state
        if hasattr(self, 'custom_step_set_validactions'):
            self.custom_step_set_validactions()
        # draw new observation
        if hasattr(self, 'draw_observation'):
            self.draw_observation(self._param_state['names'][self._state])

    def _check_validactions(self, action, valid_actions = None):
        if valid_actions is None:
            return True
        else:
            valid_actions = W.enlist(valid_actions)
            is_valid = action in valid_actions
            return is_valid

    def _advance_time(self):
        is_done = False
        self._time_state += self._time_unit
        self._time_trial += self._time_unit
        self._time_block += self._time_unit
        self._time_task += self._time_unit
        if self._time_task >= self._param_gym['n_maxTime']:
            is_done = True
        return is_done
    
    def _advance_trial(self):
        id_done = False
        self._count_block_trial += 1
        self._count_task_trial += 1
        if self._count_task_trial >= self._param_gym['n_maxTrials']:
            is_done = True
        else:
            if self._count_block_trial >= self._param_gym['block_n_maxTrials']:
                is_done = is_done or self._advance_block()
            else:
                self._reset_trial()

    def _advance_block(self):
        is_done = False
        self._count_block += 1
        if self._count_block >= self._param_gym['n_maxBlocks']:
            is_done = True
        else:
            self._reset_block()
        return is_done

    def _abort(self):
        self._trial_is_error = True
        reward = self._param_rewards['R_error']
        if "ITI" in self._param_state['names']:
            self._go_to_state("ITI")
        else:
            is_done = self._advance_trial()
        return reward, is_done
    
    def _go_to_state(self, statename):
        self._state = self._get_state_ID(statename)
        self._reset_state()
    
    def _get_state_ID(self, statename):
        return np.where([j == statename for j in self._param_state['names']])[0][0]

    def step(self, action):
        reward = 0
        # advance time
        is_done = self._advance_time()
        # transform actions
        if hasattr(self, "transform_actions"):
            action = self.transform_actions(action)
        # check valid actions
        is_error = not self._check_validactions(action, self._state_valid_actions)
        is_effective = self._check_validactions(action, self._state_effective_actions)
        # record current action
        self._record({'action': action, 'is_error': is_error})
        # get consequences of actions (reward)
        if not is_error and hasattr(self, 'custom_step_reward'):
            treward = self.custom_step_reward(action)
            reward += treward
        # get consequences of actions (state transition)
        # determine if state-transition occurs
        is_transition = False
        if not is_error:
            if self._param_state['immediate_advance'][self._state] == 1 and is_effective:
                is_transition = True
            if self._time_state >= self._param_state['timelimits'][self._state]:
                is_transition = True
        # state transition
        if is_error:
            treward, t_is_done = self._abort()
        elif is_transition:
            treward, t_is_done = self._state_transition()
        reward += treward
        is_done = is_done or t_is_done
        # get consequences of actions (after possible state transition)
        if hasattr(self, 'custom_step_reward_newstate'):
            treward = self.custom_step_reward_newstate(action)
            reward += treward
        self._record({'reward': reward})
        obs = self._get_obs()
        return obs, reward, is_done, self._time_task

    def get_probabilistic_reward(self, p):
        if np.random.uniform() <= p:
            reward = self._param_rewards['R_reward']
        else:
            reward = 0
        return reward

    def _record(self, datadict):
        pass

























    def _get_obs(self, option_augment_obs = None):
        if option_augment_obs is None:
            option_augment_obs = self._param_gym['option_augment_obs']

        if self.is_faltten_obs or self.is_augment_obs:
            obs = self.obs.flatten()
        else:
            obs = self.obs
        if (is_augment is None and self.is_augment_obs) or is_augment:
            action = self._action_flattened()
            r = np.array(self.last_reward)
            r = r.reshape((1,))
            obs = np.concatenate((obs, r, action))
        return obs
    








    def setup_stagenames(self, state_names, state_timelimits = None, \
                   state_immediateadvance = None):
        if self._param_task['is_ITI'] and not "ITI" in state_names:
            state_names.append("ITI")
        self._param_state['names'] = state_names
        self._param_state['n'] = len(self._param_state['names'])
        if state_timelimits is None:
            state_timelimits = np.Inf * np.ones(self._param_state['n']) * self._time_unit
        elif state_timelimits == "auto":
            state_timelimits = np.ones(self._param_state['n']) * self._time_unit
        self._param_state['timelimits'] = state_timelimits
        c = np.zeros(self._param_state['n'])
        if state_immediateadvance is not None:
            tid = np.array([np.where([j == i for j in self._param_state['names']]) for i in iter(state_immediateadvance)]).squeeze()
            c[tid] = 1
        self._param_state['immediate_advance'] = c

    def setup_autostatetransitions(self):
        pass






    
    def _state_transition(self):
        is_error = self._state_transition_auto()
        if not is_error:
            reward = self._param_rewards['R_advance']
            self.timer = 0 # auto reset timer
            if self.stage == len(self.metadata_stage['stage_names']):
                is_nexttrial = 1


    
        
        

        