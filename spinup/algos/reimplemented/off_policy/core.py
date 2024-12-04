import torch
from torch import nn
from spinup.algos.reimplemented.utils import MLP, _to_tensor

class CategoricalMLPQNet(nn.Module):
    def __init__(self, obs_dim, n_possible_actions, hidden_sizes, activation=nn.Tanh):
        super().__init__()

        self.net = MLP(sizes=[obs_dim, *hidden_sizes, n_possible_actions], activation=activation)

    def forward(self, obs):
        return self.net(_to_tensor(obs))
