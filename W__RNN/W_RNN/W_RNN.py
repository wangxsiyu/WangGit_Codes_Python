import torch
import torch.nn as nn
from .W_RNN_gates import W_RNN_LSTM

class W_RNN(nn.Module):
    def __init__(self, input_len, hidden_len, gatetype = "vanilla", inittype = None, device = None,\
                 *arg, **kwarg):
        super().__init__()
        self.device = device
        self.create_RNN(gatetype, input_len, hidden_len, device = device, *arg, **kwarg)
        # self.initParamsRNN(gatetype, inittype)
        
    def create_RNN(self, gatetype, input_len, hidden_len, *arg, **kwarg):
        if gatetype == "LSTM":
            self.RNN = W_RNN_LSTM(input_len, hidden_len)  
        # if gatetype == "vanilla":
        #     self.RNN = nn.RNN(input_len, hidden_len, nonlinearity = 'relu') 
        # if gatetype == "noise":
        #     self.RNN = W_RNNgate_noise(input_len, hidden_len, device=device, *arg, **kwarg) 
        self.RNN.to(self.device)