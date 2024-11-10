import argparse
import gym
import numpy as np
import torch
from tqdm import tqdm
from spinup.algos.reimplemented.utils import Logger, setup_logger_kwargs

from itertools import accumulate

from spinup.algos.reimplemented.vpg.core import _to_tensor, MLPActorCritic


def calculate_rewards_to_go(returns: torch.tensor, gamma: float, last_value: float=0) -> torch.tensor:
    """
    Calculate the rewards to go i.e. a discounted sum over the returns which only considers datapoints in the future.
    """
    returns = returns.tolist()
    returns.append(last_value)

    # x is the previously accumulated sum; y is the element at the current index
    rwtg = list(accumulate(returns[::-1], lambda x, y: gamma * x + y))[::-1]

    # we don't need the rewards to go for the last_value
    return torch.as_tensor(rwtg[:-1], dtype=torch.float32)

def generate_trajectories(
        env: gym.Env,
        steps_per_epoch: int,
        actor: MLPActorCritic,
        gamma: float,
        obs_dim: int
) -> dict:
    observation = _to_tensor(env.reset())
    traj_start_idx = 0

    obs = torch.zeros(steps_per_epoch, obs_dim, dtype=torch.float32)
    acts = torch.zeros(steps_per_epoch, dtype=torch.float32)
    rewards = torch.zeros(steps_per_epoch, dtype=torch.float32)
    returns = torch.zeros(steps_per_epoch, dtype=torch.float32)
    log_probs = torch.zeros(steps_per_epoch, dtype=torch.float32)

    traj_lengths = []

    # Generate trajectories
    for step in range(steps_per_epoch):
        dist = actor(observation)
        action = dist.sample()

        next_obs, ret, terminated, _ = env.step(action.numpy())

        obs[step] = observation
        acts[step] = action
        returns[step] = ret
        log_probs[step] = dist.log_prob(action)

        if terminated:
            # Reached terminal state
            traj_lengths.append(step - traj_start_idx)
            traj_slice = slice(traj_start_idx, step)

            rewards[traj_slice] = calculate_rewards_to_go(returns=returns[traj_slice], last_value=0, gamma=gamma)

            observation = _to_tensor(env.reset())
            traj_start_idx = step + 1

        elif step == steps_per_epoch - 1:
            # TODO: Make sure that we handle cases correctly when the trajectory gets cut to early
            # Reached end of epoch
            traj_slice = slice(traj_start_idx, step)

            # get the last value for calculation
            dist = actor(_to_tensor(next_obs))
            action = dist.sample()

            _, last_value, _, _ = env.step(action.numpy())

            rewards[traj_slice] = calculate_rewards_to_go(returns=returns[traj_slice], last_value=last_value,
                                                          gamma=gamma)
        else:
            observation = _to_tensor(next_obs)


    return dict(rewards=rewards, returns=returns, observations=obs,  actions=acts, log_probs=log_probs, traj_lengths=np.array(traj_lengths))


def basic_vpg(
        env_fn,
        hidden_size: int,
        num_hidden_layers: int,
        steps_per_epoch: int, num_epochs: int,
        gamma: float,
        lr: float=1e-4,
        seed: int=None,
        logger_kwargs=None,
):
    """
    A very basic Vanilla Policy Gradient implementation.

    Args:
        env_fn: A function that returns a gym environment.
        hidden_size: The size of a hidden layer.
        num_hidden_layers: The number of hidden layers.
        steps_per_epoch: The number of steps per epoch.
        num_epochs: The number of epochs.
        gamma: The discount factor.
        lr: The learning rate used for policy gradient updates.
        seed: The random seed used for numpy and torch.
        logger_kwargs: A dictionary of keyword arguments to pass to the logger.
    """

    # Set the random states
    if seed is not None:
        np.random.seed(42)
        torch.manual_seed(42)

    env = env_fn()

    # Initialize the actor critic
    actor = MLPActorCritic(num_hidden_layers * [hidden_size], env.action_space, env.observation_space)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)


    # Initialize the logger (only used for saving the model and state for now)
    logger = Logger(**logger_kwargs)
    logger.setup_pytorch_saver_elements(actor)

    # Train loop
    for epoch in tqdm(range(num_epochs)):
        _res = generate_trajectories(
            env,
            steps_per_epoch,
            actor,
            gamma,
            env.observation_space.shape[0]
        )

        rewards = _res['rewards']
        traj_lengths = _res['traj_lengths']
        log_probs = _res['log_probs']

        if epoch % 10 == 0:
            print(f"\nMean Rewards: {rewards.mean():.2f}, (Std: {rewards.std():.2f})")
            print(f"Mean Trajectory Length: {traj_lengths.mean():.2f} (Std: {traj_lengths.std():.2f})")


        # Compute the loss
        optimizer.zero_grad()
        loss = - (log_probs * rewards).mean()
        loss.backward()
        optimizer.step()

    # Save the final model
    logger.save_state({"env": env}, iteration=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--num_epochs", type=int, default=800)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="basic-vpg")
    args = parser.parse_args()

    # set up logger kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)


    basic_vpg(
        env_fn=(lambda: gym.make(args.env)),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        steps_per_epoch=args.steps_per_epoch,
        num_epochs=args.num_epochs,
        gamma=args.gamma,
        seed=args.seed,
        logger_kwargs=logger_kwargs,
    )