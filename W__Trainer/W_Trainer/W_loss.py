import torch
from .W_loss_A2C import W_loss_A2C

class W_loss(W_loss_A2C):
    loss_name = None
    loss_params = None
    def __init__(self, loss, device):
        self.device = device
        self.loss_name = loss['name']
        self.loss_params = loss['params']
        self.loss_cross_entropy = torch.nn.CrossEntropyLoss()
        
    def loss(self, *arg, **kwarg):
        if self.loss_name == "A2C":
            return self.loss_A2C(*arg, **kwarg)   
        elif self.loss_name == "A2C_supervised":
            return self.loss_A2C_supervised(*arg, **kwarg)     
       