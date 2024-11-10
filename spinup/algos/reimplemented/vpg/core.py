import torch
from torch import nn
from torch.distributions.categorical import Categorical

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
            NotImplemented
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


class CategoricalActorCritic(nn.Module):
    def __init__(self, hidden_sizes, action_space, observation_space, activation):
        super().__init__()

        self.logit_net = mlp([observation_space.shape[0]] + hidden_sizes + [action_space.n], activation)

    def forward(self, obs):
        return Categorical(logits=self.logit_net(obs))

    def act(self, obs):
        # For now, I just implemented this to be compatible with the test_policy script
        return self.forward(obs).sample().numpy()