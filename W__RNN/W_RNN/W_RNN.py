import torch
import torch.nn as nn
from .W_RNN_gates import W_RNN_LSTM, W_RNN_vanilla

class W_RNN(nn.Module):
    def __init__(self, input_len, hidden_len, gatetype = "vanilla", \
                 actionlayer = 2, outputlayer = None, \
                 is_param_initial = True, device = None,\
                 *arg, **kwarg):
        super().__init__()
        self.hidden_dim = hidden_len
        self.device = device
        self.create_RNN(gatetype, input_len, hidden_len, device = device, *arg, **kwarg)
        self.actionlayer = self.create_linear(actionlayer)
        self.outputlayer = self.create_linear(outputlayer)
        self.setup_initial_parameters(gatetype, is_param_initial)
        self.initialize_iolayers()

    def initialize_iolayers(self):
        torch.nn.init.orthogonal_(self.actionlayer.weight.data)
        self.actionlayer.bias.data.fill_(0)
        torch.nn.init.orthogonal_(self.outputlayer.weight.data)
        self.outputlayer.bias.data.fill_(0)

    def setup_initial_parameters(self, gatetype, is_param_initial = True): 
        if gatetype == "LSTM":
            h0 = torch.randn(self.RNN.lstm.hidden_size, requires_grad = is_param_initial).float()
            c0 = torch.randn(self.RNN.lstm.hidden_size, requires_grad = is_param_initial).float()
            if is_param_initial:
                h0 = nn.Parameter(h0)
                c0 = nn.Parameter(c0)
            self.param_initial = [h0, c0]
        elif gatetype == "vanilla":
            h0 = torch.randn(self.RNN.rnn.hidden_size, requires_grad = is_param_initial).float()
            if is_param_initial:
                h0 = nn.Parameter(h0)
            self.param_initial = [h0]

    def create_RNN(self, gatetype, input_len, hidden_len, *arg, **kwarg):
        if gatetype == "LSTM":
            self.RNN = W_RNN_LSTM(input_len, hidden_len)  
        if gatetype == "vanilla":
            self.RNN = W_RNN_vanilla(input_len, hidden_len, nonlinearity = 'relu') 
        # if gatetype == "noise":
        #     self.RNN = W_RNNgate_noise(input_len, hidden_len, device=device, *arg, **kwarg) 
        self.RNN.to(self.device)

    def create_linear(self, layer):
        if layer is not None:
            if isinstance(layer, int):
                layer = nn.Linear(self.hidden_dim, layer)
        return layer

    def forward(self, *arg, **kwarg):
        if self.training:
            return self.custom_forward(*arg, **kwarg)
        else:
            with torch.no_grad():
                return self.custom_forward(*arg, **kwarg)
            
    def get_initial_latent_value(self, batch_size = 1):
        param_initial = self.param_initial
        x = [x.repeat(batch_size,1) for x in param_initial]
        x = tuple(x)
        if len(x) == 1: # no cell vector, just hidden vector
            x = x[0]
        return x
    
    def custom_forward(self, obs, hidden_state = None):
        batch_size = obs.shape[0]
        if hidden_state is None:
            hidden_state = self.get_initial_latent_value(batch_size)   
        # obs = obs.permute((1,0,2))     
        feature, hidden_state = self.RNN(obs, hidden_state, self.device)
        self.actionlayer.to(self.device)
        actionvec = self.actionlayer(feature)
        if self.outputlayer is not None:
            self.outputlayer.to(self.device)
            outputvec = self.outputlayer(feature)
            return actionvec, hidden_state, outputvec
        else:
            return actionvec, hidden_state, None
        

