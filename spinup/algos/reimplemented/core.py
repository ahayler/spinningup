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

class A2CActor(Actor):
    def __init__(self, hidden_sizes, action_space, obs_dim, activation=nn.Tanh):
        super().__init__()

        self.backbone = MLP([obs_dim] + hidden_sizes, activation)
        self.val = nn.Linear(hidden_sizes[-1], 1)

        if isinstance(action_space, Box):
            self.pol = nn.Linear(hidden_sizes[-1], action_space.shape[0])
            self.log_std = nn.Parameter(-0.5 * torch.ones(action_space.shape[0]), requires_grad=True)
        elif isinstance(action_space, Discrete):
            self.pol = nn.Linear(hidden_sizes[-1], action_space.n)
        else:
            raise NotImplementedError

    def get_embedding(self, obs):
        return self.backbone(_to_tensor(obs))

    def get_value(self, obs):
        return self.val(self.get_embedding(obs))

    def get_distribution(self, obs):
        pol_vec = self.pol(self.get_embedding(_to_tensor(obs)))
        if hasattr(self, "log_std"):
            # continuous case
            return Normal(pol_vec, self.log_std.exp())
        else:
            # categorical/discrete case
            return Categorical(logits=pol_vec)

    def get_entropy(self, obs):
        dist = self.get_distribution(obs)

        return dist.entropy()

    def act(self, obs):
        # For now, I just implemented this to be compatible with the test_policy script
        return self.get_distribution(obs).sample().numpy()

    def log_prob_from_distribution(self, dist, act):
        if isinstance(dist, Normal):
            return dist.log_prob(_to_tensor(act)).sum(-1) # we get a log-prob per component that we then have to sum up
        elif isinstance(dist, Categorical):
            assert (len(act.shape) <= 1) | (len(act.shape) == 2 and act.shape[1] == 1)

            return dist.log_prob(_to_tensor(act).reshape(-1))
        else:
            raise NotImplementedError

    def get_log_prob_from_action(self, obs, act):
        return self.log_prob_from_distribution(self.get_distribution(obs), act)


class MLPActor(nn.Module):
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

    def get_entropy(self, obs):
        dist = self.get_distribution(obs)

        return dist.entropy()

    def get_log_prob_from_action(self, obs, action):
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

    def _check_action_dims(self, act):
        return (len(act.shape) <= 1) | (len(act.shape) == 2 and act.shape[1] == 1)

    def log_prob_from_distribution(self, dist, act):
        assert self._check_action_dims(act)

        # We need the reshape(-1) to prevent shape issues (see tests)
        return dist.log_prob(_to_tensor(act).reshape(-1))

class MLPValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()

        self.critic = MLP([obs_dim] + hidden_sizes + [1], activation)

    def get_value(self, obs):
        return self.critic(_to_tensor(obs))

    def forward(self, obs):
        return self.get_value(obs)