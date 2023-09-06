
from W_RNN.W_RNN_Gates import W_RNNgate_noise



    def initParamsRNN(self, gatetype, inittype):
        if gatetype in ["vanilla", "noise"]:
            h0 = torch.randn(self.RNN.hidden_size, requires_grad = True).float().to(self.device)
            self.h0 = nn.Parameter(h0)
            self.init0 = [self.h0]


class W_RNN_Head_ActorCritic(W_RNN):
    def __init__(self, input_len, hidden_len, action_len, gatetype = "vanilla", inittype = None, device = None, *arg, **kwarg):
        super().__init__(input_len, hidden_len, gatetype, inittype, device = device, *arg, **kwarg)
        self.actor = nn.Linear(hidden_len, action_len).to(self.device)
        self.critic = nn.Linear(hidden_len, 1).to(self.device)
        self.initParamsRNNHeadA2C(inittype)
    
    def initParamsRNNHeadA2C(self, inittype):
        self.actor.weight.data = torch.nn.init.orthogonal_(self.actor.weight.data)
        self.actor.bias.data.fill_(0)
        self.critic.weight.data = torch.nn.init.orthogonal_(self.critic.weight.data)
        self.critic.bias.data.fill_(0)

    def _forward(self, obs, hidden_state = None):
        policy_vector = self.actor(h)
        value = self.critic(h)
        return policy_vector, value, hidden_state
    