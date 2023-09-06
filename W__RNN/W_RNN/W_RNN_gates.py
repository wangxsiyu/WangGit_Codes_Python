import torch.nn as nn
# import torch

class W_RNN_LSTM(nn.Module):
    def __init__(self, input_len, hidden_len):
        super().__init__()
        self.lstm = nn.LSTM(input_len, hidden_len)

    def forward(self, input, hidden, device = None):
        (h0, c0) = hidden
        if device is not None:
            input = input.to(device)
            h0 = h0.to(device)
            c0 = c0.to(device)
            self.lstm.to(device)
        new, (h1, c1) =  self.lstm(input.unsqueeze(0), (h0.unsqueeze(0), c0.unsqueeze(0)))
        return new.squeeze(0), (h1.squeeze(0), c1.squeeze(0))