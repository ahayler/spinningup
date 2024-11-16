import argparse
import gym
import numpy as np
import torch
import time
from torch import nn
from tqdm import tqdm
from spinup.algos.reimplemented.utils import Logger, setup_logger_kwargs, discounted_cumsum

from spinup.algos.reimplemented.vpg.core import _to_tensor, MLPActorCritic, MLPValueFunction


def calculate_rewards_to_go(returns: torch.tensor, gamma: float, last_value: float=0) -> torch.tensor:
    """
    Calculate the rewards to go i.e. a discounted sum over the returns which only considers datapoints in the future.
    """
    returns = returns.tolist()
    returns.append(last_value)

    rwtg = discounted_cumsum(returns, gamma)

    # we don't need the rewards to go for the last_value
    return torch.as_tensor(rwtg[:-1], dtype=torch.float32)


def compute_gae_advantage_estimate(returns: torch.tensor, lam: float, gamma: float, vals: torch.tensor, last_value: float=0) -> torch.tensor:
    """
    Calculate the generalized advantage estimate based on the returns, the lambda & gamma hyperparameters and the predictions of our value function.
    """
    returns = np.append(returns, last_value)
    vals = np.append(vals.detach(), last_value) # we don't want gradients from the actor-critic to flow to the value function anyways

    deltas = gamma * vals[1:] - vals[:-1] + returns[:-1]

    adv = discounted_cumsum(list(deltas), lam*gamma)

    return torch.as_tensor(adv, dtype=torch.float32)

def generate_trajectories(
        env: gym.Env,
        steps_per_epoch: int,
        actor: [MLPActorCritic],
        gamma: float,
        lam: float,
        obs_dim: int,
        value_func: MLPValueFunction,
) -> dict:
    # Makes the code run faster
    with torch.no_grad():
        observation = _to_tensor(env.reset())
        traj_start_idx = 0

        obs = torch.zeros(steps_per_epoch, obs_dim, dtype=torch.float32)
        acts = torch.zeros(steps_per_epoch, dtype=torch.float32)
        rewards = torch.zeros(steps_per_epoch, dtype=torch.float32) # rewards (to-go) are used to provide supervision for the advantage function
        returns = torch.zeros(steps_per_epoch, dtype=torch.float32)
        log_probs = torch.zeros(steps_per_epoch, dtype=torch.float32)
        vals = torch.zeros(steps_per_epoch, dtype=torch.float32)
        adv = torch.zeros(steps_per_epoch, dtype=torch.float32) # the advantage estimates are used to "weigh" the log-probabilities in the loss that supervises the actor-critic

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

                # compute the values using the value function (before the gradient step)
                vals[traj_slice] = value_func(obs[traj_slice]).squeeze(1)

                rewards[traj_slice] = calculate_rewards_to_go(returns=returns[traj_slice], last_value=0, gamma=gamma)
                adv[traj_slice] = compute_gae_advantage_estimate(
                    returns=returns[traj_slice],
                    gamma=gamma,
                    vals=vals[traj_slice],
                    lam=lam,
                    last_value=0
                )

                observation = _to_tensor(env.reset())
                traj_start_idx = step + 1

            elif step == steps_per_epoch - 1:
                # Reached end of epoch
                traj_slice = slice(traj_start_idx, step)

                # get the last value for calculation
                dist = actor(_to_tensor(next_obs))
                action = dist.sample()

                _, last_value, _, _ = env.step(action.numpy())

                rewards[traj_slice] = calculate_rewards_to_go(
                    returns=returns[traj_slice],
                    last_value=last_value,
                    gamma=gamma
                )
                adv[traj_slice] = compute_gae_advantage_estimate(
                    returns=returns[traj_slice],
                    gamma=gamma,
                    vals=vals[traj_slice],
                    lam=lam,
                    last_value=last_value
                )
            else:
                observation = _to_tensor(next_obs)


    return dict(
        rewards=rewards,
        returns=returns,
        observations=obs,
        actions=acts,
        advantages=adv,
        log_probs=log_probs,
        traj_lengths=np.array(traj_lengths),
        values=vals,
    )


def basic_vpg(
        env_fn,
        hidden_size: int,
        num_hidden_layers: int,
        steps_per_epoch: int, num_epochs: int,
        gamma: float,
        lam: float,
        pol_lr: float=1e-4,
        val_lr: float=1e-3,
        gradient_steps_val_fn: int=80,
        seed: int=None,
        logger_kwargs=None,
):
    """
    A basic Vanilla Policy Gradient implementation.

    Args:
        env_fn: A function that returns a gym environment.
        hidden_size: The size of a hidden layer.
        num_hidden_layers: The number of hidden layers.
        steps_per_epoch: The number of steps per epoch.
        num_epochs: The number of epochs.
        gamma: The discount factor (used for rewards-to-go).
        lam: The discount factor (used for generalized advantage estimation)
        pol_lr: The learning rate used for policy gradient updates.
        val_lr: The learning rate used for value function updates.
        gradient_steps_val_fn: The amount of steps we do using the value function per epoch.
        seed: The random seed used for numpy and torch.
        logger_kwargs: A dictionary of keyword arguments to pass to the logger.
    """

    env = env_fn()
    start_time = time.time()

    # Set the random states
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)

    obs_dim = env.observation_space.shape[0]

    # Initialize the actor critic
    actor = MLPActorCritic(num_hidden_layers * [hidden_size], env.action_space, obs_dim)
    val_func = MLPValueFunction(obs_dim, num_hidden_layers * [hidden_size])
    loss_fn_val = nn.MSELoss()
    pol_optimizer = torch.optim.Adam(actor.parameters(), lr=pol_lr)
    val_optimizer = torch.optim.Adam(val_func.parameters(), lr=val_lr)


    # Initialize the logger (only used for saving the model and state for now)
    logger = Logger(**logger_kwargs)
    logger.setup_pytorch_saver_elements(actor)

    # Train loop
    for epoch in tqdm(range(num_epochs)):
        _res = generate_trajectories(
            env=env,
            steps_per_epoch=steps_per_epoch,
            actor=actor,
            gamma=gamma,
            lam=lam,
            obs_dim=env.observation_space.shape[0],
            value_func=val_func,
        )

        advantages = _res['advantages']
        observations = _res['observations']
        rewards = _res['rewards']
        traj_lengths = _res['traj_lengths']
        actions = _res['actions']
        #log_probs = _res['log_probs']

        # Update the actor-critic model (using a singular step)
        pol_optimizer.zero_grad()
        # before GAE: loss = - (log_probs * rewards).mean()
        # TODO: Check why no gradients get passed here to the value_function model
        # we compute log_probs again, because the trajectories are generated without gradients (to make it run faster)
        log_probs = actor.get_log_prob_for_action(observations, actions)
        loss_pol = -(log_probs * advantages).mean()
        loss_pol.backward()
        pol_optimizer.step()

        # Update the value function
        loss_val_list = []
        for _ in range(gradient_steps_val_fn):
            val_optimizer.zero_grad()
            _values = val_func(observations).squeeze(1)
            loss_val = loss_fn_val(_values, rewards)
            loss_val_list.append(loss_val.item())
            loss_val.backward()
            val_optimizer.step()

        loss_val_list = np.array(loss_val_list)

        if epoch % 10 == 0:
            elapsed_seconds = time.time() - start_time
            logger.info("") # makes the logs more readable
            logger.info(f"Epoch: {epoch}, (Time: {int(elapsed_seconds // 60):}:{elapsed_seconds % 60:02.0f})")
            logger.info(f"Mean Rewards: {rewards.mean():.2f}, (Std: {rewards.std():.2f})")
            logger.info(f"Mean Estimated Advantage: {advantages.mean():.2f}, (Std: {advantages.std():.2f})")
            logger.info(f"Mean Value Function Loss: {loss_val_list.mean():.2f}, (Std: {loss_val_list.std():.2f})")
            logger.info(f"Last Value function Loss: {loss_val_list[-1]:.2f}")
            logger.info(f"Mean Trajectory Length: {traj_lengths.mean():.2f} (Std: {traj_lengths.std():.2f})")


    # Save the final model
    logger.save_state({"env": env}, iteration=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Acrobot-v1")
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--num_epochs", type=int, default=800)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="test")
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
        lam=args.lam,
        seed=args.seed,
        logger_kwargs=logger_kwargs,
    )