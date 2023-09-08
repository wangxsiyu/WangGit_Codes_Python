from .W_RNN import W_RNN

class W_RNN_ActorCritic(W_RNN):
    def __init__(self, input_len, hidden_len, *arg, **kwarg):
        super().__init__(input_len, hidden_len, outputlayer = 1, *arg, **kwarg)
