
from W_RNN.W_RNN_Gates import W_RNNgate_noise



    def initParamsRNN(self, gatetype, inittype):
        if gatetype in ["vanilla", "noise"]:
            h0 = torch.randn(self.RNN.hidden_size, requires_grad = True).float().to(self.device)
            self.h0 = nn.Parameter(h0)
            self.init0 = [self.h0]


    