import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Config


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, bn=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn = bn
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        if self.bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, bn=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn = bn
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.fc1(state)
        if self.bn:
            x = self.bn1(x)
        xs = F.relu(x)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ContinuousActorCriticNet(nn.Module):
    """
    Actor Crictic Network (Policy) Model for continuous actions
    based on ShangtongZhang (https://github.com/ShangtongZhang/DeepRL) 
    implementation.
    """
    def __init__(self, state_size, action_size, seed, 
                 fc1_act=256, fc2_act=128, 
                 fc1_crit=256, fc2_crit=128,
                 bn=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            seed (int): Random seed.
            fc1_act (int): Actor Network Number of nodes in first hidden layer.
            fc2_act (int): Actor Network Number of nodes in second hidden layer.
            fc1_crit (int): Critic Network Number of nodes in first hidden layer.
            fc2_crit (int): Critic Network Number of nodes in second hidden layer.
            bn (bool): Batch Normalization if True.
        """
        super(ContinuousActorCriticNet, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.actor_body = Actor(state_size, 64, seed, 
                                 fc1_act, fc2_act, bn)
        self.critic_body = Actor(state_size, 64, seed, 
                                 fc1_crit, fc2_crit, bn)
        
        self.fc_action = nn.Linear(64, action_size)
        self.fc_critic = nn.Linear(64, 1)
        self.std = nn.Parameter(torch.zeros(action_size))
        
        self.reset_parameters() # Initialize model weights

    def reset_parameters(self):
        ''' Initialize Model Weights'''
        self.fc_action.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_critic.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action=None):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=Config.DEVICE)
        
        a = self.actor_body(state)
        v = self.critic_body(state)
        a = torch.tanh(self.fc_action(a)) # predicted actions
        v = self.fc_critic(v) # predicted state value
        
        dist = torch.distributions.Normal(a, F.softplus(self.std)) # Make Normal distribution sampler for predicted actions
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1) # Calculate Log probability of sampled/gived action
        entropy = dist.entropy().sum(-1).unsqueeze(-1) # Calculate entropy of action predictions
        return {'action': action,
                'log_prob': log_prob,
                'entropy': entropy,
                'a': a,
                'v': v}