import torch.nn as nn
# import torch

class W_RNN_LSTM(nn.LSTM):
    def __init__(self, input_len, hidden_len):
        super().__init__(input_len, hidden_len)

    def forward(self, input, hidden):
        super().forward(input, hidden)