from .W_RNN import W_RNN

class W_RNN_ActorCritic(W_RNN):
    def __init__(self, input_len = None, hidden_len = None, info_model = None, *arg, **kwarg):
        if info_model is None:
            assert input_len is not None
            assert hidden_len is not None
            super().__init__(input_len, hidden_len, outputlayer = 1, *arg, **kwarg)
        else:
            env = info_model['env']
            input_len = env.get_n_obs()
            hidden_len = info_model['mem-units']
            output_len = env.get_n_actions(is_motor = True)
            super().__init__(input_len, hidden_len, \
                                    gatetype = info_model['gatetype'], \
                                    actionlayer = output_len, \
                                    outputlayer = 1, *arg, **kwarg)

