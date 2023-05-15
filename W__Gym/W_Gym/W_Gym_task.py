from W_Python import W_tools as W
import numpy as np
import gym
import pandas as pd

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
    _param_task = {}
    _param_block = {}
    _param_trial = {}
    _param_state = {'n': 0, 'names': None, 'timelimits': None, \
                    'is_immediate_advance': None, 'is_terminating_state': None, \
                    'matrix_transition': None} 
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
    _last_action = None
    _last_reward = 0
    _env_vars = {} # environment moment-by-moment variables
    # draw observation
    _next_screen = None # to draw next screen
    _obs_channelID = None # input dimension vs name, for easy drawing
    # trial variables
    _trial_is_error = False
    # state variables
    _state = 0 # current state
    _state_valid_actions = None # valid actions for a state
    _state_effective_actions = None # effective actions for a state
    # data variable (recorded behavior)
    _data_issave = False
    _data_istable = True
    _data = None
    
    def __init__(self, n_maxTime = np.Inf, n_maxTrials = np.Inf, n_maxBlocks = np.Inf, \
                    block_n_maxTrials = np.Inf, option_augment_obs = ["action", "reward"], \
                    is_flatten_obs = True, is_ITI = True, dt = 1, is_save = False, is_save_table = True, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        _param_inputs = W.W_dict_kwargs()
        W.W_dict_updateonly(self._param_gym, _param_inputs)
        if self._param_gym['option_augment_obs'] is not None:
            self._param_gym['is_flatten_obs'] = True
        self._time_unit = dt
        self._data_issave = is_save
        self._data_istable = is_save_table
        if self._data_issave:
            self._data = None if self._data_istable else  {'task': None, 'block': None, 'trial': None, 'data': None}

    def setup_state_parameters(self, state_names, state_timelimits = None, \
                    state_immediateadvance = None, \
                    matrix_state_transition = "next", \
                    terminating_state = None):
        if self._param_gym['is_ITI'] and not "ITI" in state_names:
            state_names.append("ITI")
        self._param_state['names'] = state_names
        self._param_state['n'] = len(self._param_state['names'])
        # set time limits for each state
        if state_timelimits is None:
            state_timelimits = np.Inf * np.ones(self._param_state['n']) * self._time_unit
        elif state_timelimits == "auto":
            state_timelimits = np.ones(self._param_state['n']) * self._time_unit
        self._param_state['timelimits'] = state_timelimits
        # set state flag for advancing upon effective actions
        c = np.zeros(self._param_state['n'])
        if state_immediateadvance is not None:
            tid = np.array([np.where([j == i for j in self._param_state['names']]) for i in iter(state_immediateadvance)]).squeeze()
            c[tid] = 1
        self._param_state['is_immediate_advance'] = c
        # set state flag for terminating state
        if terminating_state is None:
            terminating_state = state_names[-1] # default to be last state
        terminating_state = W.enlist(terminating_state)
        c = np.zeros(self._param_state['n'])
        tid = np.array([np.where([j == i for j in self._param_state['names']]) for i in iter(terminating_state)]).squeeze()
        c[tid] = 1
        self._param_state['is_terminating_state'] = c
        # set state transition
        # if matrix_state_transition == "next":
        #     matrix_state_transition = np.zeros([self._param_state['n'], self._param_state['n']])
        #     for i in range(1, self._param_state['n']):
        #         matrix_state_transition[i-1, i] = 1
        self._param_state['matrix_transition'] = matrix_state_transition 

    def setup_reward(self, **kwarg):
        W.W_dict_updateonly(self._param_rewards, kwarg)

    def reset(self):
        self._time_task = 0
        self._count_block = 0
        self._task_trial = 0
        if hasattr(self, 'custom_reset'):
            self.custom_reset()
        self._record('task', self._param_task)
        self._reset_block()
        # draw new observation
        if hasattr(self, 'draw_observation'):
            self.draw_observation()
        # render
        if hasattr(self, 'render'):
            self.render(option = ['obs', 'time'])
        obs = self._get_obs()
        return obs
    
    def _reset_block(self):
        self._param_block = {}
        self._count_block_trial = 0
        self._time_block = 0
        if hasattr(self, 'custom_reset_block'):
            self.custom_reset_block()
        self._record('block', self._param_block)
        self._reset_trial()

    def _reset_trial(self):
        self._param_trial = {}
        self._time_trial = 0
        self._trial_is_error = False
        if hasattr(self, 'custom_reset_trial'):
            self.custom_reset_trial()
        self._state = 0
        self._record('trial', self._param_trial)
        self._reset_state()
        
    def _reset_state(self):
        self._data_state = None
        self._time_state = 0
        self._state_valid_actions = None
        self._state_effective_actions = None
        # set valid actions for the new state
        if hasattr(self, 'custom_step_set_validactions'):
            self.custom_step_set_validactions()

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
        is_done = False
        self._count_block_trial += 1
        self._count_task_trial += 1
        if self._count_task_trial >= self._param_gym['n_maxTrials']:
            is_done = True
        else:
            if self._count_block_trial >= self._param_gym['block_n_maxTrials']:
                is_done = is_done or self._advance_block()
            else:
                self._reset_trial()
        return is_done

    def _advance_block(self):
        is_done = False
        self._count_block += 1
        if self._count_block >= self._param_gym['n_maxBlocks']:
            is_done = True
        else:
            self._reset_block()
        return is_done

    def _abort(self):
        is_done = False
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

    def step(self, action_motor, is_record = True, render_options = ["obs","action","reward","time"]):
        reward = 0
        # advance time
        is_done = self._advance_time()
        tdata = {'time_task': self._time_task, 'time_trial': self._time_trial, 'state': self._param_state['names'][self._state], 'obs': self.format_obs_for_save(self._obs)}
        # transform actions
        if hasattr(self, "transform_actions"):
            action = self.transform_actions(action_motor)
        else:
            action = action_motor
        # check valid actions
        is_error = not self._check_validactions(action, self._state_valid_actions)
        is_effective = self._check_validactions(action, self._state_effective_actions)
        # get consequences of actions (reward)
        if not is_error and hasattr(self, 'custom_step_reward'):
            treward = self.custom_step_reward(action)
            reward += treward
        # get consequences of actions (state transition)
        # determine if state-transition occurs
        is_transition = False
        if not is_error:
            if self._param_state['is_immediate_advance'][self._state] == 1 and is_effective:
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
        
        # recursive component: may take multiple steps (collapse some states)
        if self._param_state['timelimits'][self._state] == 0:
            _, treward, t_is_done = self.step(action_motor, is_record = False, \
                                              render_options = None)
            reward += treward
            is_done = is_done or t_is_done
        
        self._last_action = action
        self._last_reward = reward
        tdata.update({'action': action, 'is_error': is_error, 'reward': reward})
        # record current action
        if is_record:
            self._record('data', tdata)
        # draw new observation
        if hasattr(self, 'draw_observation'):
            self.draw_observation()
        # render
        if hasattr(self, 'render'):
            obs_renderer = self.render(option = render_options)
        if hasattr(self, 'metadata_render') and self.metadata_render['render_mode'] == "rgb_array":
            obs = obs_renderer
        else:
            obs = self._get_obs()
        return obs, reward, is_done, self._time_task

    def _state_transition(self):
        reward = 0
        is_done = False
        if self._param_state['is_terminating_state'][self._state]:
            is_done = self._advance_trial()
        else:
            if hasattr(self, 'custom_state_transition'):
                self._state = self.custom_state_transition()
            elif self._param_state['matrix_transition'] == "next":
                self._state += 1
            else:
                transprob = self._param_state['matrix_transition'][self._state]
                self._state = np.random.choice(np.arange(0, self._param_state['n']), 1, p=transprob)
        return reward, is_done
    
    def get_probabilistic_reward(self, p):
        if np.random.uniform() <= p:
            reward = self._param_rewards['R_reward']
        else:
            reward = 0
        return reward

    def _get_obs(self):
        option_augment_obs = self._param_gym['option_augment_obs']
        obs = self._obs.flatten() if self._param_gym['is_flatten_obs'] else self._obs
        if option_augment_obs is not None:
            for opt_name in iter(option_augment_obs):
                if opt_name == "action": 
                    tval = self._get_action_onehot(self._last_action)
                elif opt_name == "reward":
                    tval = np.array(self._last_reward)
                    tval = tval.reshape((1,))
                obs = np.concatenate((obs, tval))
        return obs
    
    def format_obs_for_save(self, obs):
        return obs.flatten()
    
    def _get_action_onehot(self, action):
        n = self.get_n_actions()
        action_onehot = np.zeros(n)
        if action is not None:
            action_onehot[action] = 1
        return action_onehot
    
    def get_n_actions(self):
        return self.action_space.n

    def get_n_obs(self, is_count_augmented_dimensions = True):
        if self.observation_space.shape == ():
            len = self.observation_space.n
        else:
            len = self.observation_space.shape
        if is_count_augmented_dimensions and self._param_gym['option_augment_obs'] is not None:
            for opt_name in iter(self._param_gym['option_augment_obs']):
                if opt_name == "action": 
                    len += self.get_n_actions()
                elif opt_name == "reward":
                    len += 1
        return len                

    def flip(self, is_clear = True):
        self._obs = self._next_screen
        if is_clear:
            self.blankscreen()

    def blankscreen(self):
        assert hasattr(self, 'observation_space')
        self._next_screen = np.zeros(self.get_n_obs(is_count_augmented_dimensions=False))

    # draw obs
    def draw_onehot(self, channelname, val):
        if self._next_screen is None:
            self.blankscreen()
        if channelname == "ITI":
            return
        assert self._obs_channelID is not None
        idx = self._obs_channelID[channelname]
        self._next_screen[idx] = val

    def setup_obs_channelID(self, mydict):
        self._obs_channelID = mydict
    
    def _record(self, datatype, datadict = None):
        if not self._data_issave:
            return
        if self._data_istable and not datatype == "data":
            return
        if self._data_istable: # must have datatype == "data"
            datadict.update(self._param_trial)
            datadict.update(self._param_block)
            datadict.update(self._param_task)
            df = self._data 
            if df is None:
                df = pd.DataFrame()
            self._data = pd.concat((df, pd.DataFrame.from_dict(datadict, orient = "index").T))
        else:        
            df = self._data[datatype] 
            if df is None:
                df = pd.DataFrame()
            self._data[datatype] = pd.concat((df, pd.DataFrame.from_dict(datadict, orient = "index").T))
    







    


    
        
        

        