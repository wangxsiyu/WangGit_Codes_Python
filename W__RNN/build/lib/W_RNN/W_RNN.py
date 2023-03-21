import torch
import torch.nn as nn

class W_RNN(nn.Module):
    def __init__(self, input_len, hidden_len, gatetype = "vanilla", inittype = None):
        super().__init__()
        self.makeRNN(gatetype, input_len, hidden_len)
        self.initParamsRNN(gatetype, inittype)

    def makeRNN(self, gatetype, input_len, hidden_len):
        if gatetype == "LSTM":
            self.RNN = nn.LSTM(input_len, hidden_len)    
    
    def initParamsRNN(self, gatetype, inittype):
        if gatetype == "LSTM":
            h0 = torch.randn(1, 1, self.RNN.hidden_size).float()
            c0 = torch.randn(1, 1, self.RNN.hidden_size).float()
            self.h0 = nn.Parameter(h0)
            self.c0 = nn.Parameter(c0)

    def get_h0_c0(self):
        return (self.h0, self.c0)

class W_RNN_Head_ActorCritic(W_RNN):
    def __init__(self, input_len, hidden_len, action_len, gatetype = "vanilla", inittype = None):
        super().__init__(input_len, hidden_len, gatetype, inittype)
        self.actor = nn.Linear(hidden_len, action_len)
        self.critic = nn.Linear(hidden_len, 1)
        self.initParamsRNNHeadA2C(inittype)
    
    def initParamsRNNHeadA2C(self, inittype):
        self.actor.weight.data = torch.nn.init.orthogonal_(self.actor.weight.data)
        self.actor.bias.data.fill_(0)
        self.critic.weight.data = torch.nn.init.orthogonal_(self.critic.weight.data)
        self.critic.bias.data.fill_(0)

    def forward(self, obs, hidden_state = None):
        if hidden_state is None:
            hidden_state = self.get_h0_c0()        
        h, hidden_state = self.RNN(obs, hidden_state)
        action_dist = torch.nn.functional.softmax(self.actor(h), dim = -1)
        value = self.critic(h)
        return action_dist, value, hidden_state

