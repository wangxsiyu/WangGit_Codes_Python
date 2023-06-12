import torch.nn as nn
import torch

class W_RNNgate_noise(nn.Module):
    def __init__(self, input_len, hidden_len, noise_scale = 0, device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.noise_scale = noise_scale
        super().__init__()
        self.hidden_size = hidden_len
        self.layer_encode = nn.Linear(input_len + hidden_len, hidden_len, device = self.device)
        
    def forward(self, input, hidden):
        input = input.to(self.device)
        out = []
        for i in range(input.shape[0]):
            combined = torch.cat((input[i,:,:].unsqueeze(0), hidden), dim = -1)
            noise  = self.get_noise(combined.shape)
            hidden = nn.functional.relu(self.layer_encode(combined + noise))
            out.append(hidden)
        return (torch.cat(out)), hidden
    
    
    def get_noise(self, shape):
        """get Gaussian noise"""
        noise = torch.normal(mean=0, std=1, size=shape).to(self.device)
        return self.noise_scale * noise