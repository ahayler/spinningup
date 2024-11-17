import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np

from gymnasium.spaces import Box, Discrete


def _to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.Tanh):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)])
        self.activation = activation()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        return self.layers[-1](x)


class Actor(nn.Module):
    def get_distribution(self, obs):
        raise NotImplementedError

    def log_prob_from_distribution(self, dist, act):
        raise NotImplementedError


class MLPActorCritic(nn.Module):
    def __init__(self, hidden_sizes, action_space, obs_dim, activation=nn.Tanh):
        super().__init__()

        if isinstance(action_space, Box):
            self.actor = GaussianActorCritic(hidden_sizes, action_space, obs_dim, activation)
        elif isinstance(action_space, Discrete):
            self.actor = CategoricalActorCritic(hidden_sizes, action_space, obs_dim, activation)

    def forward(self, obs):
        return self.get_distribution(obs)

    def get_distribution(self, obs):
        return self.actor.get_distribution(obs)

    def get_log_prob_for_action(self, obs, action):
        dist = self.get_distribution(obs)

        return self.actor.log_prob_from_distribution(dist, action)

    def act(self, obs):
        # For now, I just implemented this to be compatible with the test_policy script
        return self.actor.get_distribution(obs).sample().numpy()


class GaussianActorCritic(Actor):
    def __init__(self, hidden_sizes, action_space, obs_dim, activation):
        super().__init__()

        act_dim = action_space.shape[0]

        # One advantage of using log is that our parameters can be unbounded
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim, dtype=torch.float32), requires_grad=True)

        self.mu_net = MLP(sizes=[obs_dim] + hidden_sizes + [act_dim], activation=activation)

    def get_distribution(self, obs):
        mu = self.mu_net(_to_tensor(obs))

        return Normal(mu, self.log_std.exp())

    def log_prob_from_distribution(self, dist, act):
        return dist.log_prob(_to_tensor(act)).sum(dim=-1) # the log prob of the distribution, is the sum of the individual logprobs

class CategoricalActorCritic(Actor):
    def __init__(self, hidden_sizes, action_space, obs_dim, activation):
        super().__init__()

        self.logit_net = MLP([obs_dim] + hidden_sizes + [action_space.n], activation)

    def get_distribution(self, obs):
        return Categorical(logits=self.logit_net(_to_tensor(obs)))

    def log_prob_from_distribution(self, dist, act):
        return dist.log_prob(_to_tensor(act))

class MLPValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()

        self.critic = MLP([obs_dim] + hidden_sizes + [1], activation)

    def forward(self, obs):
        return self.critic(_to_tensor(obs))