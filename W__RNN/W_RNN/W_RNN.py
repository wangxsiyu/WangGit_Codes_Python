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
        if gatetype == "vanilla":
            self.RNN = nn.RNN(input_len, hidden_len, nonlinearity = 'relu')  
    
    def initParamsRNN(self, gatetype, inittype):
        if gatetype == "LSTM":
            self.h0 = torch.randn(self.RNN.hidden_size).float()
            self.c0 = torch.randn(self.RNN.hidden_size).float()
            self.h0.requires_grad = True
            self.c0.requires_grad = True
            self.init0 = [self.h0, self.c0]
        if gatetype == "vanilla":
            self.h0 = torch.randn(self.RNN.hidden_size).float()
            self.h0.requires_grad = True
            self.init0 = [self.h0]

    def get_h0_c0(self, batch_size = 1):
        init0 = self.init0
        x = [x.repeat(1, batch_size,1) for x in init0]
        x = tuple(x)
        if len(x) == 1:
            x = x[0]
        return x

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

    def _forward(self, obs, hidden_state = None):
        if hidden_state is None:
            batch_size = obs.shape[0]
            hidden_state = self.get_h0_c0(batch_size)   
        obs = obs.permute((1,0,2))     
        h, hidden_state = self.RNN(obs, hidden_state)
        action_dist = torch.nn.functional.softmax(self.actor(h), dim = -1)
        value = self.critic(h)
        return action_dist, value, hidden_state
    
    def forward(self, *arg, **kwarg):
        if self.training:
            return self._forward(*arg, **kwarg)
        else:
            with torch.no_grad():
                return self._forward(*arg, **kwarg)