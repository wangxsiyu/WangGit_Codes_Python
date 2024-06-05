    

    def get_auto_savename(self):
        if hasattr(self, 'custom_savename'):
            tstr = self.custom_savename()
            savename = f"{self.env_name}_{tstr}"
        else:
            savename = f"{self.env_name}"
        return savename

    def setup_obs_dim(self, *arg):
        self.dim_obs = np.array(arg)
        self.n_obs = np.prod(self.dim_obs)

    def setup_action_dim(self, n_actions, n_motor = None):
        self.n_actions = n_actions
        if n_motor is None:
            n_motor = n_actions
        self.n_motor = n_motor

    def setup_obs_channel_namedict(self, mydict):
        self._obs_channel_namedict = mydict

    def setup_state_parameters(self, state_names, state_timelimits = None, \
                    state_immediateadvance = None, \
                    matrix_state_transition = "next", \
                    terminating_state = None):
        if self._metadata_gym['is_ITI'] and not "ITI" in state_names:
            state_names.append("ITI")
        self._metadata_state['statenames'] = state_names
        self._metadata_state['n_state'] = len(self._metadata_state['statenames'])
        # set time limits for each state
        if state_timelimits is None:
            state_timelimits = np.ones(self._metadata_state['n_state']) * np.Inf
        elif state_timelimits == "ones":
            state_timelimits = np.ones(self._metadata_state['n_state']) * self._time_unit
        self._metadata_state['timelimits'] = state_timelimits
        # set state flag for advancing upon effective actions
        c = np.zeros(self._metadata_state['n_state'])
        if state_immediateadvance is not None:
            tid = self._find_state(state_immediateadvance)
            c[tid] = 1
        self._metadata_state['is_immediate_advance'] = c
        # set state flag for terminating state
        if terminating_state is None:
            if "ITI" in self._metadata_state['statenames']:
                terminating_state = "ITI"
            else:
                terminating_state = state_names[-1] # default to be last state
        terminating_state = W.W_enlist(terminating_state)
        c = np.zeros(self._metadata_state['n_state'])
        tid = self._find_state(terminating_state)
        c[tid] = 1
        self._metadata_state['is_terminating_state'] = c
        # set state transition
        # if matrix_state_transition == "next":
        #     matrix_state_transition = np.zeros([self._metadata_state['n_state'], self._metadata_state['n_state']])
        #     for i in range(1, self._metadata_state['n_state']):
        #         matrix_state_transition[i-1, i] = 1
        self._metadata_state['matrix_transition'] = matrix_state_transition 






        

    
        

    

    

    
    def get_probabilistic_reward(self, p):
        if np.random.uniform() <= p:
            reward = self._param_rewards['R_reward']
        else:
            reward = 0
        return reward

    
    

    def get_n_obs(self, is_count_augmented_dimensions = True):
        #if self.observation_space.shape == ():
        #    len = self.observation_space.n
        #else:
        #    len = self.observation_space.shape
        
        assert hasattr(self, 'n_obs')
        len = self.n_obs
        
        if is_count_augmented_dimensions and self._metadata_gym['option_obs_augment'] is not None:
            for opt_name in iter(self._metadata_gym['option_obs_augment']):
                if opt_name == "action": 
                    len += self.get_n_actions(is_motor=False)
                elif opt_name == "motor":
                    len += self.get_n_actions(is_motor=True)
                elif opt_name == "reward":
                    len += 1
                else:
                    len += np.array(self._env_vars[opt_name]).size
        return len                
    
    def get_dim_obs(self):
        assert hasattr(self, 'dim_obs')
        return self.dim_obs

    def flip(self, is_clear = True):
        self._obs = self._next_screen
        if is_clear:
            self.blankscreen()

    def blankscreen(self):
        # assert hasattr(self, 'observation_space')
        self._next_screen = np.zeros(self.get_dim_obs())

    # draw obs
    def draw_onehot(self, channelname, val):
        if self._next_screen is None:
            self.blankscreen()
        if channelname == "ITI":
            return
        assert self._obs_channel_namedict is not None
        idx = self._obs_channel_namedict[channelname]
        self._next_screen[idx] = val
    
    
