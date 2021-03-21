import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Dict

# run on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    """Implements the Actor (Policy) network.

    According to 'Supplementary Material' section 'C' in above paper network
    layout is

        (state dim, 400)
        ReLU
        (400, 300)
        ReLU
        (300, action dim)
        tanh
    """

    def __init__(self, state_size: int, action_size: int, action_high: float,
                 conf: Dict) -> None:

        """Creates the Actor network layers."""
        super(Actor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high

        self.lin1 = nn.Linear(self.state_size, 400)
        self.lin2 = nn.Linear(400, 300)
        self.lin3 = nn.Linear(300, self.action_size)
        self.action_high = torch.tensor(self.action_high)

        self.init_weights(conf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the forward activation functions."""
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.action_high * torch.tanh(self.lin3(x))
        return x

    def init_weights(self, conf: Dict) -> None:
        """uses the callable stores in `conf` to initialize network weights"""
        if 'INIT_FN_ACTOR' in conf:
            initializer = conf['INIT_FN_ACTOR']
        else:
            initializer = init.xavier_uniform_

        for module in self.modules():
            if type(module) == nn.Linear:
                if type(initializer) == init.uniform_:
                    b = conf.get('INIT_W_MAX_ACTOR', 0.005)
                    a = conf.get('INIT_W_MIN_ACTOR', -b)
                    initializer(module.weight.data, a=a, b=b)
                elif type(initializer) == init.constant_:
                    val = conf.get('INIT_W_ACTOR', 0.)
                    initializer(module.weight.data, val)
                else:
                    initializer(module.weight.data)


class Critic(nn.Module):
    """Implements the Critic (Value) networks.

    According to 'Supplementary Material' section 'C' in above paper network
    layout is

        (state dim + action dim, 400)
        ReLU
        (400, 300)
        RelU
        (300, 1)

    """

    def __init__(self, state_size: int, action_size: int, conf: Dict) -> None:
        """Creates a new Critic network."""
        super(Critic, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.lin1 = nn.Linear(self.state_size + self.action_size, 400)
        self.lin2 = nn.Linear(400, 300)
        self.lin3 = nn.Linear(300, 1)

        self.init_weights(conf)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Adds the forward activation functions."""
        xu = torch.cat((x, u), 1)
        x1 = F.relu(self.lin1(xu))
        x1 = F.relu(self.lin2(x1))
        x1 = self.lin3(x1)
        return x1

    def init_weights(self, conf: Dict) -> None:
        """uses the callable stores in `conf` to initialize network weights"""
        if 'INIT_FN_CRITIC' in conf:
            initializer = conf['INIT_FN_CRITIC']
        else:
            initializer = init.xavier_uniform_

        for module in self.modules():
            if type(module) == nn.Linear:
                if type(initializer) == init.uniform_:
                    b = conf.get('INIT_W_MAX_CRITIC', 0.005)
                    a = conf.get('INIT_W_MIN_CRITIC', -b)
                    initializer(module.weight.data, a=a, b=b)
                elif type(initializer) == init.constant_:
                    val = conf.get('INIT_W_CRITIC', 0.)
                    initializer(module.weight.data, val)
                else:
                    initializer(module.weight.data)
