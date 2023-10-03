import torch.nn as nn
# import torch

class W_RNN_LSTM(nn.Module):
    def __init__(self, input_len, hidden_len):
        super().__init__()
        self.lstm = nn.LSTM(input_len, hidden_len)

    def forward(self, input, hidden, device = None):
        (h0, c0) = hidden # batch size x units
        h0 = h0.unsqueeze(0) # 1 (D * num_layers) x batch size x dimensions
        c0 = c0.unsqueeze(0)
        if len(input.shape) == 2: # unbatched
            input = input.unsqueeze(0) # batch x sequence len x dimensions
        input = input.permute((1,0,2)) # L x batch x dimensions
        if device is not None:
            input = input.to(device)
            h0 = h0.to(device)
            c0 = c0.to(device)
            self.lstm.to(device)
        new, (h1, c1) =  self.lstm(input, (h0, c0))
        new = new.permute((1,0,2))
        # return new, (h1, c1)
        return new.squeeze(0), (h1.squeeze(0), c1.squeeze(0))
    
    def get_latent_units(self, LV):
        return LV[0]
    
class W_RNN_vanilla(nn.Module):
    def __init__(self, input_len, hidden_len, *arg, **kwarg):
        super().__init__()
        self.rnn = nn.RNN(input_len, hidden_len, *arg, **kwarg)

    def forward(self, input, hidden, device = None):
        h0 = hidden.unsqueeze(0) # 1 (D * num_layers) x batch size x dimensions
        if len(input.shape) == 2: # unbatched
            input = input.unsqueeze(0) # batch x sequence len x dimensions
        input = input.permute((1,0,2)) # L x batch x dimensions
        if device is not None:
            input = input.to(device)
            h0 = h0.to(device)
            self.rnn.to(device)
        new, h1 =  self.rnn(input, h0)
        new = new.permute((1,0,2))
        return new.squeeze(0), h1.squeeze(0)
    