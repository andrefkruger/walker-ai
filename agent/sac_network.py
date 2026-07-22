import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import config as C

LOG_STD_MIN = -5
LOG_STD_MAX = 2


def _init(layer, std=1.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0)
    return layer


def _mlp(in_dim, hidden_dim, hidden_layers, out_dim, out_std=1.0):
    layers = []
    d = in_dim
    for _ in range(hidden_layers):
        layers += [_init(nn.Linear(d, hidden_dim)), nn.ReLU()]
        d = hidden_dim
    layers.append(_init(nn.Linear(d, out_dim), std=out_std))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Política estocástica com re-parametrização (squashed Gaussian)."""

    def __init__(self):
        super().__init__()
        self.net     = _mlp(C.STATE_DIM, C.HIDDEN_DIM, C.HIDDEN_LAYERS,
                            C.HIDDEN_DIM, out_std=1.0)
        self.mean    = _init(nn.Linear(C.HIDDEN_DIM, C.ACTION_DIM), std=0.01)
        self.log_std = _init(nn.Linear(C.HIDDEN_DIM, C.ACTION_DIM), std=0.01)

    def forward(self, state):
        x       = F.relu(self.net(state))  # net já tem ReLU no final, mas ok
        mean    = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std  = log_std.exp()
        dist = Normal(mean, std)
        x    = dist.rsample()                    # re-parametrização
        action = torch.tanh(x)

        # log prob com correção de tanh
        log_prob = dist.log_prob(x) \
                   - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class Critic(nn.Module):
    """Twin Q-networks (Q1 e Q2) para reduzir overestimation."""

    def __init__(self):
        super().__init__()
        in_dim = C.STATE_DIM + C.ACTION_DIM
        self.q1 = _mlp(in_dim, C.HIDDEN_DIM, C.HIDDEN_LAYERS, 1, out_std=1.0)
        self.q2 = _mlp(in_dim, C.HIDDEN_DIM, C.HIDDEN_LAYERS, 1, out_std=1.0)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
