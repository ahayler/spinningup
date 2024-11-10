import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from gym.spaces import Box, Discrete


def _to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


class mlp(nn.Module):
    def __init__(self, sizes, activation=nn.Tanh):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)])
        self.activation = activation()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        return self.layers[-1](x)


class MLPActorCritic(nn.Module):
    def __init__(self, hidden_sizes, action_space, observation_space, activation=nn.Tanh):
        super().__init__()

        if isinstance(action_space, Box):
            self.actor = GaussianActorCritic(hidden_sizes, action_space, observation_space, activation)
        elif isinstance(action_space, Discrete):
            self.actor = CategoricalActorCritic(hidden_sizes, action_space, observation_space, activation)

    def forward(self, obs):
        return self.actor.forward(obs)

    def act(self, obs):
        # For now, I just implemented this to be compatible with the test_policy script
        return self.actor.forward(obs).sample().numpy()


class GaussianActorCritic(nn.Module):
    def __init__(self, hidden_sizes, action_space, observation_space, activation):
        super().__init__()

        act_dim = action_space.shape[0]

        # One advantage of using log is that our parameters can be unbounded
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim, dtype=torch.float32), requires_grad=True)

        self.mu_net = mlp(sizes=[observation_space.shape[0]] + hidden_sizes + [act_dim], activation=activation)

    def forward(self, obs):
        mu = self.mu_net(_to_tensor(obs))

        return Normal(mu, self.log_std.exp())



class CategoricalActorCritic(nn.Module):
    def __init__(self, hidden_sizes, action_space, observation_space, activation):
        super().__init__()

        self.logit_net = mlp([observation_space.shape[0]] + hidden_sizes + [action_space.n], activation)

    def forward(self, obs):
        return Categorical(logits=self.logit_net(obs))