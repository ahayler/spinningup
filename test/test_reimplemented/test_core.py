import pytest
from spinup.algos.reimplemented.core import MLPActor, A2CActor
from torch import nn
import gymnasium as gym
import torch

def test_categorical_actor_critic_shape_issues():
    """
    This test tries to prevent a very hard to detect bug, where the log_probs vector for categorical variables
    is not as expected of shape (batch_dim, 1), but instead of shape (batch_dim, batch_dim). Loss computation and
    everything else still runs through. Just convergence becomes far less stable.
    """
    env = gym.make('CartPole-v0')

    obs_dim = env.observation_space.shape[0]

    mlp_actor = MLPActor([5, 5], env.action_space, obs_dim, activation=nn.Tanh)
    ac2_actor = A2CActor([5, 5], env.action_space, obs_dim, activation=nn.Tanh)

    test_actions = torch.zeros(100)
    test_obs = torch.zeros(100, obs_dim)

    mlp_log_probs_squeezed = mlp_actor.get_log_prob_from_action(test_obs, test_actions)
    mlp_log_probs_unsqueezed = mlp_actor.get_log_prob_from_action(test_obs, test_actions.unsqueeze(-1))

    assert mlp_log_probs_squeezed.shape == mlp_log_probs_unsqueezed.shape

    a2c_log_probs_squeezed = ac2_actor.get_log_prob_from_action(test_obs, test_actions)
    a2c_log_probs_unsqueezed = ac2_actor.get_log_prob_from_action(test_obs, test_actions.unsqueeze(-1))

    assert a2c_log_probs_squeezed.shape == a2c_log_probs_unsqueezed.shape
